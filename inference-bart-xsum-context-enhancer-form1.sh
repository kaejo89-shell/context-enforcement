export CUDA_VISIBLE_DEVICES=2
export PYTHONPATH=.:$PYTHONPATH
nohup accelerate launch  context_enforcement/trainers/inference.py \
--run-id bart-base-context-enhance-form1 \
--trained-model-path trained-models/  >> logs/inference-bart-base-context-enhance-form1.txt &