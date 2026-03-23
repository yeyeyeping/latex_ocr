#!/opt/conda/bin/python3
#coding=utf-8
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
import os
os.environ['PYTORCH_ALLOC_CONF'] = "expandable_segments:True"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "expandable_segments:True"
import gc
import json
from torch.nn import functional as F
import torch
from transformers.optimization import get_cosine_schedule_with_warmup
from args import parse_args,c2net_context
from c2net.context import upload_output
import bitsandbytes as bnb
from utils import configure_logger,backup_code, configure_tensorboard,print_trainable_parameters,read_gpu_info,manual_seed,set_tf32,save_model
from pathlib import Path
from dataset import build_dataloader
from models import build_models
from tqdm import tqdm
def train_for_iterations(step, total_steps, gradient_accumulation_steps, max_iteration, model, train_iter, optimizer, scheduler):
    model.train()
    tbar = tqdm(range(total_steps), desc=f"TRAIN [{step}/{max_iteration}]")
    sum_grad_norm = 0.0
    sum_loss = 0.0
    for _ in tbar:
        minstep_loss = 0.0
        for _ in range(gradient_accumulation_steps):
            data_dict = next(train_iter)            
            input_ids = data_dict.pop('input_ids').cuda()                                                                               
            pixel_values = data_dict.pop('pixel_values').cuda()
            attention_mask = data_dict.pop('attention_mask').cuda()
            labels = data_dict.pop('labels').cuda()
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                
                out_dict = model(input_ids=input_ids,
                                pixel_values=pixel_values, 
                                attention_mask=attention_mask,
                                use_cache=False)
                
                logits = out_dict.logits
                
                logits = logits[:, :-1, :].contiguous()
                labels = labels[:, 1:].contiguous()
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
                loss = loss / gradient_accumulation_steps
                
            loss.backward()
            minstep_loss += loss.item()
            del out_dict, logits, labels, input_ids, pixel_values, attention_mask, loss
        step_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0).item()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        tbar.set_postfix(loss=minstep_loss, grad_norm=step_grad_norm)
        
        sum_loss += minstep_loss
        sum_grad_norm += step_grad_norm
    gc.collect()
    torch.cuda.empty_cache()
    training_loss = sum_loss / total_steps
    mean_grad_norm = sum_grad_norm / total_steps    
    return {
        "step": step + total_steps,
        "training_loss": round(training_loss, 4),
        "mean_grad_norm": round(mean_grad_norm, 4),
        "lr": round(scheduler.get_last_lr()[0], 8)
    }


def config_lr(model, lr, logger):
    lora_vision_towner_param = {}
    lora_llm_towner_param = {}
    modules_to_save_param = {}
    unkown_param = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            if "lora_" in name:
                if "vision_tower" in name:
                    lora_vision_towner_param[name] = {"params": param, "lr": lr * 3}
                elif "language_model" in name:
                    lora_llm_towner_param[name] = {"params": param, "lr": lr}
                else:
                    unkown_param[name] = {"params": param, "lr": lr}
            elif "modules_to_save" in name:
                modules_to_save_param[name] = {"params": param, "lr": lr / 10}
            else:
                logger.info(f"WARNING: {name} is set to require grad, but not in LoRA or modules_to_save.")
    logger.info(f"lora in vision towner: {'|'.join([name + ':' + str(round(v['lr'], 6)) for name, v in lora_vision_towner_param.items()])}")
    logger.info(f"lora in language_model: {'|'.join([name + ':' + str(round(v['lr'], 6)) for name, v in lora_llm_towner_param.items()])}")
    logger.info(f"lora in modules_to_save: {'|'.join([name + ':' + str(round(v['lr'], 6)) for name, v in modules_to_save_param.items()])}")
    logger.info(f"lora in unkown param: {'|'.join([name + ':' + str(round(v['lr'], 6)) for name, v in unkown_param.items()])}")
    return list(lora_llm_towner_param.values()) + list(lora_vision_towner_param.values()) + list(modules_to_save_param.values()) + list(unkown_param.values())

def main():
    set_tf32()
    training_args, model_args, data_args = parse_args()
    manual_seed(training_args.seed)
    logger = configure_logger(training_args.working_dir)
    writer = configure_tensorboard(c2net_context.tensorboard_path)
    
    logger.info(f"Training arguments: {training_args}\nModel arguments: {model_args}\nData arguments: {data_args}")
    backup_code(Path(training_args.working_dir)/"code_backup")
    
    logger.info(read_gpu_info())
    
    model, processor = build_models(model_args)
    print_trainable_parameters(model)
    
    # Enable torch compile for faster training
    logger.info("Compiling model with torch.compile...")
    try:
        model = torch.compile(model)
        logger.info("Model compiled successfully.")
    except Exception as e:
        logger.warning(f"Model compilation failed: {e}")
        logger.info("Using uncompiled model.")
        
    logger.info("Building dataloader...")
    train_dataloader = build_dataloader(data_args, processor)
    global_batch_size = data_args.batch_size * training_args.gradient_accumulation_steps
    max_iteration = training_args.epoch * len(train_dataloader) // training_args.gradient_accumulation_steps
    logger.info(f"Total training iterations: {max_iteration}, gloabl batch size: {global_batch_size}")
    
    optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=training_args.lr)    #
    # set lora modules lr to training_args.lr, module_to_save to training_args.lr/10
    # optimizer_param_groups = config_lr(model, training_args.lr, logger)
    # optimizer = bnb.optim.AdamW8bit(optimizer_param_groups)    #
    # optimizer = AdamW([param for param in model.parameters() if param.requires_grad], lr=training_args.lr)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=max_iteration//5, num_training_steps=max_iteration)
    
    logger.info("Starting training...")
    
    def infinite_loader(dataloader):
        while True:
            for batch in dataloader:
                yield batch
            
    train_iterator = infinite_loader(train_dataloader)
    for step in range(0,  max_iteration, training_args.log_intervl):
        interval = min(training_args.log_intervl, max_iteration - step)
        metric_dict = train_for_iterations(
                            step,
                            interval,
                            training_args.gradient_accumulation_steps,
                            max_iteration,
                            model,
                            train_iterator,
                            optimizer,
                            scheduler)                        
        logger.info(json.dumps(metric_dict))
        
        writer.add_scalar("train/loss", metric_dict["training_loss"], metric_dict["step"])
        writer.add_scalar("train/mean_grad_norm", metric_dict["mean_grad_norm"], metric_dict["step"])
        writer.add_scalar("train/lr", metric_dict["lr"], metric_dict["step"])
        

        
        if step % training_args.save_intervl == 0 and step > 0:
            save_path = Path(training_args.working_dir)/f"checkpoint-{metric_dict['step']}"
            save_path.mkdir(parents=True, exist_ok=True)
            save_model(model, save_path, processor)
            logger.info(f"Saved checkpoint to {save_path}")
    # Final save
    save_path = Path(training_args.working_dir)/f"checkpoint-final"
    save_model(model, save_path, processor)
    
    
    
if __name__ == "__main__":
    try:
        main()
    finally:
        upload_output()
