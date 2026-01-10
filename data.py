from PIL import Image
from datasets import load_dataset
import io
import torch
from utils import bounded_resize
from transformers import InternVLProcessor
from transformers.image_utils import infer_channel_dimension_format,ChannelDimension
from transformers.image_transforms import to_channel_dimension_format,resize
from transformers.models.got_ocr2.image_processing_got_ocr2 import get_optimal_tiled_canvas
from PIL import Image
from typing import Optional, Union
import numpy as np
from PIL.Image import Resampling as PILImageResampling
import io
def crop_to_patch(
        images: np.ndarray,
        min_patches: int=1,
        max_patches: int=12,
        patch_size: Optional[Union[tuple, int, dict]] = None,
        data_format: Optional[ChannelDimension] = None
        ):
    if data_format is None:
        data_format = infer_channel_dimension_format(images)
    images = to_channel_dimension_format(images, ChannelDimension.FIRST, data_format)
    patch_size_height, patch_size_width = patch_size["height"], patch_size["width"]
    original_height, original_width = images.shape[-2:]
    # find the closest aspect ratio to the target
    num_columns, num_rows = get_optimal_tiled_canvas(
        (original_height, original_width), (patch_size_height, patch_size_width), min_patches, max_patches
    )

    # calculate the target width and height
    target_width = patch_size_width * num_columns
    target_height = patch_size_height * num_rows

    # resize the image so that each patch is of patch_size
    resized_image = resize(
        images,
        size=(target_height, target_width),
        resample=PILImageResampling.BICUBIC,
        data_format=ChannelDimension.FIRST,
        input_data_format=ChannelDimension.FIRST,
    )
    return resized_image
patch_size = {
    "height": 448,
    "width": 448,
}

def format_message(example, prompt, min_patches=1, max_patches=12):
    answer = example['text']
    image =  Image.open(io.BytesIO(example['image'])).convert("RGB")
    image = np.array(image)
    image = crop_to_patch(
            images=image,
            min_patches=min_patches,
            max_patches=max_patches,
            patch_size=patch_size,
            data_format=None,
        )
    image = Image.fromarray(np.transpose(image, (1,2,0)))
    return {
            "messages":[
                {
                    "role": "user", 
                    "content": 
                        [
                            {"type":"image"},
                            {"type":"text","text": prompt},
                        ]
                },
                    {
                        "role": "assistant", 
                        "content": 
                        [
                            {"type":"text","text": answer}
                        ]
                    }
              ],
            "images": [image]
        }


def build_dataset(args):
    dataset = load_dataset(args.data_folder)
    train_dataset,val_dataset = dataset['train'],dataset['validation']
    train_dataset = train_dataset.map(lambda x: format_message(x, args.prompt), 
                                      num_proc=4, 
                                      load_from_cache_file=False)
    val_dataset = val_dataset.map(lambda x: format_message(x,  args.prompt),
                                  num_proc=4, 
                                  load_from_cache_file=False)
    return train_dataset, val_dataset

def get_collate_fn(processor:InternVLProcessor, train_transform=None, assistant_token_id=77091):
    
    def collate_fn(examples):
        images, texts = zip(*[(example["images"][0], example['messages']) for example in examples])
        images = list(images)
        for i in range(len(images)):
            # images[i] = bounded_resize(images[i])
            if train_transform:
                images[i] = train_transform(images[i])
        texts = processor.apply_chat_template(texts, tokenize=False)
        batch = processor(text=texts, 
                          images=list(images), 
                          padding_side="right", 
                          return_tensors="pt", 
                          padding=True) 
        
    
        attention_mask = batch["attention_mask"]
        labels = batch["input_ids"].clone() 
        input_ids = batch["input_ids"]
        
        
        batch_size,seq_len = input_ids.shape
        _, y = torch.where(input_ids == assistant_token_id)
        pos = torch.arange(seq_len).repeat(batch_size, 1)
        y = y + 1
        pos[pos <= y.unsqueeze(1)] = False
        pos[pos > y.unsqueeze(1)] = True
        assistant_tokens_mask = pos.bool()
        labels[(~assistant_tokens_mask) | (~attention_mask.bool())] = -100
        return {
            "input_ids": batch["input_ids"],
            "pixel_values": batch["pixel_values"],
            "attention_mask": batch["attention_mask"],
            "labels": labels
        }
    return collate_fn