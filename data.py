from PIL import Image
from datasets import load_dataset
import io
import torch
from transformers import InternVLProcessor
def format_message(example, prompt, tgt_image_size):
    answer = example['text']
    image =  Image.open(io.BytesIO(example['image'])).convert("RGB").resize((tgt_image_size, tgt_image_size))
    return {
            "messages":[
                {
                    "role": "user", 
                    "content": 
                        [
                            {"type":"text","text": prompt},
                            {"type":"image"}
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
    train_dataset = train_dataset.map(lambda x: format_message(x, args.prompt, args.tgt_image_size), num_proc=8)
    val_dataset = val_dataset.map(lambda x: format_message(x,  args.prompt, args.tgt_image_size), num_proc=8)
    return train_dataset, val_dataset

def get_collate_fn(processor:InternVLProcessor, train_transform=None, assistant_token_id=77091):
    
    def collate_fn(examples):
        images, texts = zip(*[(example["images"][0], example['messages']) for example in examples])
        if train_transform:
            images = [train_transform(x) for x in images]
        
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