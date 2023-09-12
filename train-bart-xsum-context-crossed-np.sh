export CUDA_VISIBLE_DEVICES=2
export PYTHONPATH=.:$PYTHONPATH
nohup accelerate launch  context_enforcement/trainers/train_bart.py \
--from-scratch \
--max-seq-len 800 \
--context-max-len 250 \
--task-type xsum \
--num-train-epochs 15 \
--eval-steps 1000 \
--lr-scheduler-type linear \
--learning-rate 0.0005 \
--warmup-ratio 0.30 \
--weight-decay 0.01 \
--per-device-train-batch-size 16 \
--per-device-eval-batch-size 16 \
--save-total-limit 3 \
--model-base  facebook/bart-base \
--run-id bart-base-from-scratch-context-crossed \
--fp16 \
--gradient-accumulation-steps 4 \
--output-dir trained-models/  >> logs/bart-base-from-scratch-context-crossed.out &