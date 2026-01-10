
# ckpt_path=/root/project/code/ckpt/20260105_125755/checkpoint-610
prompt='请从输入的图像中还原LaTeX 代码,直接输出latex公式,并保证输出的latex代码语法正确：'
# /root/.conda/envs/vllm/bin/vllm serve $ckpt_path --served_model_name internvl --port 8000
/root/.conda/bin/python /root/project/latex_ocr-main/run_eva.py "$prompt"
/root/.conda/bin/python /root/project/VLM-formula-recognition-dataset-main/main_eval.py
