import peft

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
        lora_alpha=lora_rank * 2,
        target_modules = tgt_modules,
    )

    model = peft.get_peft_model(model, cfg)
    return model

