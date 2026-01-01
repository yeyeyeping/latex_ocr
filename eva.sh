
ckpt_path=/root/project/code/ckpt/20260101_162217/checkpoint-280
prompt='你是一位专业的 LaTeX 排版专家，擅长从数学公式图像中精准还原高质量的 LaTeX 代码.请根据图片中的公式生成对应的 latex 公式文本：'
/root/.conda/envs/vllm/bin/vllm serve $ckpt_path --served_model_name internvl --port 8000
/root/.conda/bin/python /root/project/latex_ocr/evaluate.py $prompt
/root/.conda/bin/python /root/project/VLM-formula-recognition-dataset-main/main_eval.py

