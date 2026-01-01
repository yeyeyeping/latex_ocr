import time
from pathlib import Path
from PIL import Image
from io import BytesIO
import base64
import openai
from tqdm import tqdm
import sys

def format_message(image_bytes, prompt):
    return [
                {
                    "role": "user", 
                    "content": 
                        [
                            {"type":"text","text": prompt},
                            {"type":"image_url", "image_url":{"url":f'data:image/jpeg;base64,{base64.b64encode(image_bytes).decode("utf-8")}'}}
                        ]
                },
              ]

def pil_image2_bytes(image:Image.Image)->bytes:
    image_io = BytesIO()
    image.save(image_io, format='PNG')
    image_bytes = image_io.getvalue()
    return image_bytes

prompt = sys.argv[1]    



valid_image_path = Path("/root/project/hf/datasets/train/val_images")
out_latex_dir = Path("predictions")
out_latex_dir.mkdir(parents=True, exist_ok=True)


client = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="")
for img_path in tqdm(valid_image_path.glob("*.png")):
    image = Image.open(img_path).convert("RGB")
    image_bytes = pil_image2_bytes(image)
    message = format_message(image_bytes, prompt)
    resp = client.chat.completions.create(
        model="internvl",
        messages=message,
        temperature=0,
        max_tokens=1024)
    latex_code = resp.choices[0].message.content
    out_file = out_latex_dir / f"{img_path.stem}.txt"
    with open(out_file, "w") as f:
        f.write(latex_code)
    
    
    
    
    