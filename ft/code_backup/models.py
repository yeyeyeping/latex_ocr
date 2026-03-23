import math
import peft
from transformers import InternVLForConditionalGeneration, InternVLProcessor
import torch
import logging
def inject_lora(model, lora_rank,lora_alpha, 
                vision_lora=None, 
                llm_lora=None, 
                lora_mm_project=False, 
                lm_head=False):
    tgt_modules = ""
    if llm_lora and llm_lora != "":
        tgt_modules = f"(model\.language_model\..*\.({llm_lora}))"
    
    if vision_lora and vision_lora != "":
        tgt_modules += f"|(model\.vision_tower\..*\.({vision_lora}))"
    modules_to_save = []

    if lora_mm_project:
        modules_to_save.append("multi_modal_projector")
    if lm_head:
        modules_to_save.append("embed_tokens")
        modules_to_save.append("lm_head")
        
    cfg = peft.LoraConfig(
        r = lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        use_rslora = True, 
        
        target_modules = tgt_modules,
        modules_to_save=modules_to_save,
        )

    model = peft.get_peft_model(model, cfg)
    return model

def build_models(model_args):
    logger = logging.getLogger("__main__")
    # 初始化模型
    processor = InternVLProcessor.from_pretrained(model_args.model_path, trust_remote_code=True)

    try:
        model = InternVLForConditionalGeneration.from_pretrained(
            model_args.model_path,
            device_map="auto",
            attn_implementation="flash_attention_2",
            dtype=torch.bfloat16, trust_remote_code=True
        )
        logger.info("using flash_attention_2")
    except Exception:
        model = InternVLForConditionalGeneration.from_pretrained(
            model_args.model_path,
            device_map="auto",
            dtype=torch.bfloat16, 
            trust_remote_code=True
        )
        logger.info("using default attention")
    processor.image_processor.max_patches = 12
    logger.info(f"model: {type(model)}, tie word embeddings: {model.config.tie_word_embeddings}")
    logger.info(f"processor: {type(processor)}, image_processor: {type(processor.image_processor)}, max_patches: {processor.image_processor.max_patches}")
    # untie weights
    model.config.tie_word_embeddings = False
    model._tied_weights_keys = []
    new_weights = model.language_model.get_input_embeddings().weight.clone()
    model.lm_head.weight = torch.nn.Parameter(new_weights)
    
    
    model = inject_lora(
        model,
        model_args.lora_rank,
        model_args.lora_alpha,
        model_args.tgt_module_in_vision,
        model_args.tgt_module_in_llm,
        model_args.mm_projector,
        model_args.lm_head
    )
    
    # LoRA 注入后再设置，与 TRL 一致
    model.enable_input_require_grads()

    # 使用 use_reentrant=False 以兼容 torch.compile
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    
    model.config.use_cache = False
    model.vision_tower.use_cache = False
    model.language_model.use_cache = False
    
    return model, processor