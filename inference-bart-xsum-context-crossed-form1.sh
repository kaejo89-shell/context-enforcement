export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH=.:$PYTHONPATH
nohup accelerate launch --main_process_port=8000 context_enforcement/trainers/inference.py \
--run-id bart-base-context-crossed-form1 \
--is-form2 \
--trained-model-path trained-models/  >> logs/inference-bart-base-context-crossed-form1.txt &