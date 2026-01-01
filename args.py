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
    prompt: str = """你是一位专业的 LaTeX 排版专家，擅长从数学公式图像中精准还原高质量的 LaTeX 代码.请根据图片中的公式生成对应的 latex 公式文本：
                    输出格式要求：
                    1. 必须使用 ```latex 代码块包裹
                    2. 仅包含 LaTeX 代码，无任何文字说明
                    3. 确保语法正确，可以通过编译，所有下标用 {} 括起来

                    输出案例：
                    案例 1：
                    ```latex
                    \sum_{i=1}^{n} x_i = \mu
                    ```

                    案例 2：
                    ```latex
                    \begin{bmatrix}
                    a & b \\
                    c & d
                    \end{bmatrix}
                    ```
                    """


def parse_args()->Tuple[TrainingArguments,ModelArgument,DataArgument]:
    parser = HfArgumentParser((TrainingArguments, ModelArgument , DataArgument))
    training_args, model_args, data_args = parser.parse_args_into_dataclasses()

    return training_args, model_args, data_args
