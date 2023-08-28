export CUDA_VISIBLE_DEVICES=3
export PYTHONPATH=.:$PYTHONPATH
nohup python context_enforcement/trainers/train_bart3.py \
--is-enforcement-baseline \
--max-seq-len 800 \
--task-type xsum \
--num-train-epochs 10 \
--eval-steps 1000 \
--lr-scheduler-type polynomial \
--learning-rate 3e-5 \
--warmup-ratio 0.10 \
--weight-decay 0.01 \
--per-device-train-batch-size 12 \
--per-device-eval-batch-size 12 \
--save-total-limit 1 \
--model-base  facebook/bart-base \
--run-id bart-base-xsum-context-enforcer-baseline \
--fp16 \
--gradient-accumulation-steps 4 \
--output-dir trained_models/xsum/  >> logs/training_logs_bart_base_xsum-context-enforcer-baseline.out &