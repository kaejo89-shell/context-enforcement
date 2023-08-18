export CUDA_VISIBLE_DEVICES=2
export PYTHONPATH=.:$PYTHONPATH
nohup python context_enforcement/trainers/train_t5.py \
--context-max-len 200 \
--max-seq-len 512 \
--task-type xsum \
--num-train-epochs 10 \
--eval-steps 1000 \
--lr-scheduler-type linear \
--learning-rate 4e-3 \
--warmup-ratio 0.25 \
--per-device-train-batch-size 12 \
--per-device-eval-batch-size 12 \
--save-total-limit 1 \
--model-base  t5-base \
--run-id t5-base_model_ce2 \
--gradient-accumulation-steps 4 \
--output-dir trained_models/t5models/  >> logs/training_logs_t5-base_model_context_enf4.out &
