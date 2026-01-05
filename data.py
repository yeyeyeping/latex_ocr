from PIL import Image
from datasets import load_dataset
import io
import torch
from utils import bounded_resize
from transformers import InternVLProcessor
def format_message(example, prompt):
    answer = example['text']
    image =  Image.open(io.BytesIO(example['image'])).convert("RGB")
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
                          padding_side="left", 
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