from pathlib import Path
from dataclasses import dataclass
from time import localtime, strftime
from transformers import HfArgumentParser
from typing import Tuple
from c2net.context import prepare
c2net_context = prepare()
@dataclass
class TrainingArguments:
    lr: float = 1e-4
    epoch: int = 5
    gradient_accumulation_steps: int = 8
    seed: int=43
    save_intervl:int = 400
    log_intervl:int = 10

@dataclass
class ModelArguments:
    model_path:str = c2net_context.pretrain_model_path + "/internvl3.5_2"
    lora_rank: int = 128
    lora_alpha:int = 64
    tgt_module_in_vision: str = ""
    tgt_module_in_llm:str = "(q_proj|k_proj|v_proj|o_proj|down_proj|up_proj|gate_proj)"
    lm_head:bool = True
    mm_projector:bool = True

@dataclass
class DataArguments:
    parquet_file: str = c2net_context.dataset_path + "/latex_ocr_linxy_2w/selected_final.parquet"
    prompt: str = """OCR:"""
    batch_size: int = 8

def parse_args()->Tuple[TrainingArguments,ModelArguments,DataArguments]:
    parser = HfArgumentParser((TrainingArguments, ModelArguments , DataArguments))
    training_args, model_args, data_args = parser.parse_args_into_dataclasses()
    training_args.working_dir = Path(c2net_context.output_path)
    return training_args, model_args, data_args

