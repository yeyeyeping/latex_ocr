from dataclasses import dataclass
from transformers import HfArgumentParser
from typing import Tuple
@dataclass
class TrainingArguments:
    lr: float = 2e-4
    epoch: int = 10
    batch_size: int = 8
    gradient_accumulation_steps: int = 8
    seed: int=43
    

@dataclass
class ModelArgument:
    model_path:str = "/root/project/hf/models/InternVL3_5-HF"
    lora_rank: int = 32
    tgt_module_in_vision:str = None
    tgt_module_in_llm:str = "(q_proj|k_proj|v_proj|o_proj|down_proh|up_proj)"
    mm_projector:bool = False

@dataclass
class DataArgument:
    data_folder = "/root/project/code/data/data/intern"
    tgt_image_size: int = 448
    prompt: str = """请从输入的图像中还原LaTeX 代码,保证输出的latex代码语法正确："""


def parse_args()->Tuple[TrainingArguments,ModelArgument,DataArgument]:
    parser = HfArgumentParser((TrainingArguments, ModelArgument , DataArgument))
    training_args, model_args, data_args = parser.parse_args_into_dataclasses()

    return training_args, model_args, data_args
