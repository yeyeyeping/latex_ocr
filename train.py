from transformers import InternVLForConditionalGeneration, InternVLProcessor,Qwen3ForCausalLM
import torch
from pathlib import Path
import time
from trl import SFTConfig,SFTTrainer
from transformers.trainer_callback import EarlyStoppingCallback,PrinterCallback
from transformers import set_seed
from torchvision.transforms import (GaussianBlur, ColorJitter, RandomRotation, Compose, RandomApply, RandomPerspective,RandomAffine) 
from utils import read_gpu_info
from args import parse_args
from data import get_collate_fn
import shutil
from models import inject_lora,length_balaced_ce
from data import build_dataset
import json
from utils import print_trainable_parameters, configure_logger,backup_code
from callback import LoggingCallback
def get_eval_dataloader(self, eval_dataset=None):
    eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
    return torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=self.args.eval_batch_size,
        collate_fn=get_collate_fn(self.processing_class),
    )


training_args, model_args, data_args = parse_args()
set_seed(training_args.seed)

out_dir = Path(f"working_dir/{time.strftime('%Y%m%d_%H%M%S')}")
out_dir.mkdir(parents=True, exist_ok=True)
# 保存代码文件
cur_dir = Path(__file__).parent

backup_code(out_dir / "code")

logger = configure_logger(out_dir)
# 打印GPU信息
logger.info(json.dumps(read_gpu_info(), ensure_ascii=False))

logger.info(json.dumps({
        "training_args":str(training_args),
        "model_args": str(model_args),
        "data_args": str(data_args)
    }, ensure_ascii=False))
# 初始化模型
processor = InternVLProcessor.from_pretrained(model_args.model_path)
try:
    model = InternVLForConditionalGeneration.from_pretrained(model_args.model_path, attn_implementation="flash_attention_2",dtype=torch.bfloat16)
    logger.info("using flash_attention_2")
except:
    model = InternVLForConditionalGeneration.from_pretrained(model_args.model_path, dtype=torch.bfloat16)
model.loss_function = length_balaced_ce
model = inject_lora(model,
            model_args.lora_rank, 
            model_args.tgt_module_in_vision,
            model_args.tgt_module_in_llm,
            model_args.mm_projector,
            model_args.lm_head)

print_trainable_parameters(model)
train_dataset, val_dataset = build_dataset(data_args)

# Configure training arguments
training_args = SFTConfig(
    output_dir=out_dir,  
    num_train_epochs=training_args.epoch,
    per_device_train_batch_size=training_args.batch_size,  
    gradient_accumulation_steps=training_args.gradient_accumulation_steps,  
    
    per_device_eval_batch_size=2 ,
    eval_accumulation_steps=training_args.gradient_accumulation_steps,
    gradient_checkpointing=True,  
    
    eval_strategy = "steps",
    optim="adamw_torch_fused",
    learning_rate=training_args.lr,  
    weight_decay=0.001,
    lr_scheduler_type="cosine",  
    logging_steps=5,
    warmup_ratio=0.1,
    
    eval_steps = 10,
    metric_for_best_model="eval_mean_token_accuracy",
    greater_is_better=True,
    
    torch_empty_cache_steps = 10,
    save_strategy="best",
    save_only_model=True,
    save_total_limit = 3, 
    
    dataloader_pin_memory = True,
    dataloader_num_workers = 8,
    dataloader_persistent_workers = True,
    dataloader_prefetch_factor = 2,
    
    bf16=True,
    torch_compile=True,
    report_to="tensorboard",
    max_grad_norm=1,
    gradient_checkpointing_kwargs={"use_reentrant": False},   
)

# MEAN = [int(i * 255) for i in processor.image_processor.image_mean]
train_transform = Compose(
     [
        RandomApply([GaussianBlur((3,3))]),
        RandomApply([RandomPerspective(distortion_scale=0.4, p=1,fill=255),
                     RandomRotation(degrees=(-10, 10), fill=255)]),
        RandomApply([ColorJitter(brightness=0.5, contrast=0.5)]),
        RandomApply([RandomAffine(degrees=0, scale=(0.8, 1.2), fill=255)], p=0.3)  
     ])


trainer = SFTTrainer(
    model=model,
    data_collator=get_collate_fn(processor, train_transform),
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    processing_class=processor,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=15), LoggingCallback(logger)]
)
# remove PrinterCallback
trainer.remove_callback(PrinterCallback)
trainer.get_eval_dataloader = get_eval_dataloader.__get__(trainer)

trainer.train()


trainer.save_model(out_dir/"latest")