from torchvision.transforms.v2.functional import pad
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, RandomApply, ColorJitter, RandomPerspective, RandomRotation, RandomAffine, GaussianBlur, Pad
import torch
from torch.utils.data import Dataset
from io import BytesIO
from PIL import Image
import pyarrow.parquet as pq
from torchvision.transforms.functional import pil_to_tensor
from torch.nn import functional as F
from torch import Tensor, nn
from torchvision.io import decode_jpeg, encode_jpeg
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="PIL")
MEAN_FILL = [255, 255, 255]
class SFTDataset(Dataset):
    def __init__(self, 
                 processor,
                 prompt:str,
                 parquet_file, 
                 transform=None
                 ):
        super().__init__()
        self.data = pq.read_table(parquet_file, memory_map=True, use_threads=False)
        self.transform = transform
        self.processor = processor
        self.prompt = prompt
        self.assistant_token_id = processor.tokenizer.convert_tokens_to_ids("assistant")
    
    def __len__(self):
        return self.data.shape[0]
    
    def bytes2tensor(self, image_bytes):
        read_io = BytesIO(image_bytes)
        with Image.open(read_io) as img:
                pil_image = img.convert("RGB")
        read_io.close()
        return pil_to_tensor(pil_image)
    
    def format_message(self, latex_str):
        return [
                {
                    "role": "user", 
                    "content": 
                        [
                            {"type":"image"},
                            {"type":"text","text": self.prompt},
                        ]
                },
                    {
                        "role": "assistant", 
                        "content": 
                        [
                            {"type":"text","text": latex_str}
                        ]
                    }
              ]
    
    def find_assistant_token_idx(self, input_ids):                                                                                                                                              
        mask = input_ids == self.assistant_token_id                                                                                                                                             
        if not mask.any():                                                                                                                                                                      
            raise ValueError("Assistant token not found")                                                                                                                                       
        return mask.nonzero(as_tuple=True)[0][0].item() 
    
    def letterbox_resize(self, image: Tensor, target_size: int) -> Tensor:
        _, h, w = image.shape
        scale = target_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        resized_image = F.interpolate(image.unsqueeze(0).float(), size=(new_h, new_w), mode='bilinear', align_corners=False).squeeze(0).to(image.dtype)
        padded_image = torch.tensor(MEAN_FILL, dtype=image.dtype).view(3,1,1).repeat(1,target_size,target_size)
        pad_left = (target_size - new_w) // 2
        pad_top = (target_size - new_h) // 2
        padded_image[:, pad_top:pad_top+new_h, pad_left:pad_left+new_w] = resized_image
        return padded_image
    
    def __getitem__(self, index):
        image_bytes = self.data.column("image")[index].as_py()
        latex_str = self.data.column("text")[index].as_py()
        
        image_tensor = self.bytes2tensor(image_bytes)
        # print(image_tensor.shape)
        image_tensor = self.transform(image_tensor) if self.transform else image_tensor
        # image_tensor = self.letterbox_resize(image_tensor, target_size=448)
        chat_latex_str = self.processor.apply_chat_template(self.format_message(latex_str), tokenize=False)
        out_dict = self.processor(
                            images=image_tensor, 
                            text=chat_latex_str,
                            return_tensors="pt")
        # remove batch dimension
        input_ids = out_dict["input_ids"][0]
        assistant_token_idx = self.find_assistant_token_idx(input_ids)
        return {
            "input_ids": input_ids,
            "pixel_values": out_dict["pixel_values"], # keep num of patches dimension
            "attention_mask": out_dict["attention_mask"][0],# remove batch dimension
            "assistant_token_idx": torch.as_tensor([assistant_token_idx], dtype=torch.long)
        }
        
class RandomApplyJpeg(nn.Module):
    def __init__(self, min_quality: int, max_quality: int) -> None:
        assert 1 <= min_quality <= max_quality <= 100
        super().__init__()
        self.min_quality = min_quality
        self.max_quality = max_quality

    def forward(self, x: Tensor) -> Tensor:
        quality = torch.randint(self.min_quality, self.max_quality + 1, ()).item()
        return decode_jpeg(encode_jpeg(x, quality))


class RandomShift(nn.Module):
    def __init__(self, max_shift: int, fill=255):
        super().__init__()
        self.max_shift = max_shift
        self.fill = fill

    def forward(self, image: Tensor) -> Tensor:
        _, h, w = image.shape
        shift_x = torch.randint(-self.max_shift, self.max_shift + 1, ()).item()
        shift_y = torch.randint(-self.max_shift, self.max_shift + 1, ()).item()
        shifted_image = pad(image, (self.max_shift, self.max_shift, self.max_shift, self.max_shift),fill=self.fill)
        shifted_image = shifted_image[:, self.max_shift + shift_y : self.max_shift + shift_y + h, self.max_shift + shift_x : self.max_shift + shift_x + w]
        return shifted_image
    
    
def get_collate_fn(pad_id):
    def collate_fn(batch):    
        max_len = max([item['input_ids'].shape[0] for item in batch])
        # padded to max_len
        padded_input_ids_list, padded_attention_mask_list = [],[]
        for item in batch:
            input_ids = item['input_ids']
            attention_mask = item['attention_mask']
            pad_length = max_len - input_ids.shape[0]
            if pad_length > 0:
                input_ids = F.pad(input_ids, (0, pad_length), value=pad_id)
                attention_mask = F.pad(attention_mask, (0, pad_length), value=0)
            padded_input_ids_list.append(input_ids)
            padded_attention_mask_list.append(attention_mask)
        
        padded_input_ids = torch.stack(padded_input_ids_list, dim=0)
        padded_attention_mask = torch.stack(padded_attention_mask_list, dim=0)

        # SFT only compute loss on assistant response        
        assistant_token_idx = torch.stack([item['assistant_token_idx'] for item in batch], dim=0)
        prefix_mask = torch.arange(0, max_len).unsqueeze(0).repeat(len(batch), 1)
        # the position <= assistant_token_idx + 1 are prefix, note that the +1 is to include the \n following assistant token
        prefix_mask = (prefix_mask <= assistant_token_idx + 1).bool()
        labels = torch.where(prefix_mask | (~padded_attention_mask.bool()), -100, padded_input_ids)
        
        pixel_values = torch.cat([item['pixel_values'] for item in batch], dim=0)
        return {
            "input_ids": padded_input_ids,
            "pixel_values": pixel_values,
            "attention_mask": padded_attention_mask,
            "labels": labels
        }
    return collate_fn        


    
def build_dataloader(data_args, processor):

    train_transform = Compose(
     [
        RandomApply([GaussianBlur((3,3))]),
        RandomApply([RandomPerspective(distortion_scale=0.2, p=1,fill=MEAN_FILL),
                     RandomRotation(degrees=(-4, 4), fill=MEAN_FILL)]),
        RandomApply([ColorJitter(brightness=0.3, contrast=0.3)]),
        RandomApply([RandomAffine(degrees=0, scale=(0.8, 1.2), fill=MEAN_FILL)], p=0.3),
        RandomApplyJpeg(min_quality=30, max_quality=100),
        RandomApply([RandomShift(max_shift=15,fill=MEAN_FILL)],)
     ])
    
    dataset = SFTDataset(processor=processor,
               prompt=data_args.prompt,
               parquet_file=data_args.parquet_file,
               transform=train_transform)
    collate_fn = get_collate_fn(processor.tokenizer.pad_token_id)
    
    dataloader = DataLoader(dataset, 
                            batch_size=data_args.batch_size, 
                            shuffle=True,
                            persistent_workers=True,
                            pin_memory=False,
                            num_workers=12,
                            prefetch_factor=2,
                            collate_fn=collate_fn)
    
    return dataloader