export CUDA_VISIBLE_DEVICES=2
export PYTHONPATH=.:$PYTHONPATH
nohup accelerate launch context_enforcement/trainers/train_bart.py \
--is-baseline \
--max-seq-len 800 \
--task-type wikihow \
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
--run-id bart-base-baseline \
--fp16 \
--gradient-accumulation-steps 6 \
--output-dir trained-models/wikihow/  >> logs/wikihow/training_logs_bart_base_model_baseline.txt &