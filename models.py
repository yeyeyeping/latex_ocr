import evaluate
from torch import nn
import peft
import torch
import numpy as np
import os

os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"
def inject_lora(model, lora_rank, vision_lora=None, llm_lora=None, lora_mm_project=False, lm_head=False):
    tgt_modules = ""
    if llm_lora and llm_lora != "":
        tgt_modules = f"(model\.language_model\..*\.({llm_lora}))"
    
    if vision_lora and vision_lora != "":
        tgt_modules += f"|(model\.vision_tower\..*\.({vision_lora}))"
    
    if lora_mm_project:
        tgt_modules += r"|(model\.multi_modal_projector\.linear_[12])"
        tgt_modules += r"|(model\.multi_modal_projector\.linear_[13])"
    
    if lm_head:
        tgt_modules += r"|(lm_head)"
    
    cfg = peft.LoraConfig(
        r = lora_rank,
        lora_dropout=0.05,
        use_rslora = True,
        target_modules = tgt_modules,
    )

    model = peft.get_peft_model(model, cfg)
    return model

def length_balaced_ce(
   logits,
    labels,
    vocab_size,
    num_items_in_batch = None,
    ignore_index = -100,
    shift_labels = None,
    **kwargs,
) :
    # Upcast to float if we need to compute the loss to avoid potential precision issues
    logits = logits.float()

    if shift_labels is None:
        # Shift so that tokens < n predict n
        labels = nn.functional.pad(labels, (0, 1), value=ignore_index)
        shift_labels = labels[..., 1:].contiguous()

    # Flatten the tokens
    logits = logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)
    # Enable model parallelism
    shift_labels = shift_labels.to(logits.device)
    loss = nn.functional.cross_entropy(logits, shift_labels, ignore_index=ignore_index, reduction="none")
    loss = loss.view(labels.size(0), -1).mean(1)
    shift_labels = shift_labels.view(labels.size(0), -1) 
    
    txt_length = (shift_labels == ignore_index).sum(1) + 1 # plus one to avoid zero division
    txt_length = torch.rsqrt(txt_length)
    txt_length = txt_length / txt_length.sum()
    return (loss * txt_length).sum()


def preprocess_logits_for_metrics(logits, labels=None):
    """Collapse logits to token ids so eval does not cache full vocab tensors."""
    if isinstance(logits, tuple):
        logits = logits[0]
    return torch.argmax(logits, dim=-1)


def get_blue_metrics(tokenizer):
    bleu = evaluate.load("bleu")

    def inner_metrics_compute(eval_preds):
        preds, labels = eval_preds.predictions, eval_preds.label_ids
        if isinstance(preds, tuple):
            preds = preds[0]
        preds_flat = np.asarray(preds)
        labels_flat = np.asarray(labels)
        decoded_preds, decoded_labels = [], []
        for p, g in zip(preds_flat, labels_flat):
            mask = g != -100
            valid_pred = p[mask]
            valid_gt = g[mask]
            decoded_preds.append(tokenizer.decode(valid_pred, skip_special_tokens=True))
            decoded_labels.append(tokenizer.decode(valid_gt, skip_special_tokens=True))
        
        bleu_result = bleu.compute(predictions=decoded_preds, references=decoded_labels)
        return {
            "eval_bleu": bleu_result["bleu"],
        }

    return inner_metrics_compute