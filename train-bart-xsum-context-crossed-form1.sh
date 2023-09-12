export CUDA_VISIBLE_DEVICES=2
export PYTHONPATH=.:$PYTHONPATH
nohup accelerate launch --main_process_port=8201   context_enforcement/trainers/train_bart_crossed_form1.py \
--max-seq-len 800 \
--context-max-len 310 \
--task-type xsum \
--num-train-epochs 15 \
--eval-steps 1000 \
--lr-scheduler-type linear \
--learning-rate 0.00006 \
--warmup-ratio 0.35 \
--weight-decay 0.01 \
--per-device-train-batch-size 8 \
--per-device-eval-batch-size 12 \
--save-total-limit 3 \
--model-base  facebook/bart-base \
--run-id bart-base-context-crossed-form1 \
--fp16 \
--gradient-accumulation-steps 6 \
--output-dir trained-models/  >> logs/bart-base-context-crossed-form1.txt &