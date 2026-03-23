from tensorboardX import SummaryWriter
from transformers.utils.logging import get_logger
import torch
import logging
import sys
import shutil
from pathlib import Path
from logging import FileHandler,StreamHandler
def set_tf32():
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')
        torch.backends.cudnn.allow_tf32 = True

def save_model(model, save_path, processor):
    save_path.mkdir(parents=True, exist_ok=True)
    if hasattr(model, "module"):
        model.module.save_pretrained(save_path)
    else:
        model.save_pretrained(save_path)
    processor.save_pretrained(save_path)

def read_gpu_info():
    gpu_info = {}
    gpu_info['cuda_available'] = torch.cuda.is_available()
    gpu_info['gpu_count'] = torch.cuda.device_count()
    gpu_info['current_device_index'] = torch.cuda.current_device()
    gpu_info['gpu_name'] = torch.cuda.get_device_name(torch.cuda.current_device())
    if torch.cuda.is_available():
        gpu_info['allocated_gpu_memory_GB'] = torch.cuda.memory_allocated() / 1024**3
        gpu_info['cached_gpu_memory_GB'] = torch.cuda.memory_reserved() / 1024**3
    
    return gpu_info

def print_trainable_parameters(model):
    logger = logging.getLogger("__main__")
    trainable_params = 0
    all_param = 0
    for name, param in model.named_parameters():
        num_params = param.numel()
        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
            logger.info(name)
            
    
    logger.info(f"trainable params: {trainable_params:,} || all params: {all_param:,} || trainable%: {100 * trainable_params / all_param:.2f}%")
    

def manual_seed(seed):
    """Set manual seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    # no one care about cudnn deterministic, set to False for better performance
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    
def configure_tensorboard(out_dir):
    tb_writer = SummaryWriter(logdir=out_dir)
    return tb_writer
def configure_logger(out_dir):
    logger = logging.getLogger("__main__")
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M"
    )
    file_handler = FileHandler(out_dir/"log.txt", encoding="utf8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    
    console_hanler =StreamHandler(sys.stdout)
    console_hanler.setFormatter(formatter)
    console_hanler.setLevel(logging.INFO)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_hanler)
    logger.propagate = False
    return logger

def backup_code(tgt_dir):
    tgt_dir.mkdir(parents=True, exist_ok=True)
    base_dir = Path(__file__).parent
    for item in base_dir.iterdir():
        if item.name.endswith('.py',) or item.name.endswith('.sh'):
            shutil.copy(item, tgt_dir / item.name)