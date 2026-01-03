from torch import nn
import peft
import torch
def inject_lora(model, lora_rank, vision_lora=None, llm_lora=None, lora_mm_project=False):
    tgt_modules = ""
    if llm_lora and llm_lora != "":
        tgt_modules = f"(model\.language_model\..*\.({llm_lora}))"
    
    if vision_lora and vision_lora != "":
        tgt_modules += f"|(model\.vision_tower\..*\.({vision_lora}))"
    
    if lora_mm_project:
        tgt_modules += r"|(model\.multi_modal_projector\.linear_[12])"
    
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


