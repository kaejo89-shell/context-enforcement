export CUDA_VISIBLE_DEVICES=2
export PYTHONPATH=.:$PYTHONPATH
nohup accelerate launch context_enforcement/trainers/train_bart.py \
--max-seq-len 800 \
--context-max-len 250 \
--task-type xsum \
--num-train-epochs 10 \
--eval-steps 1000 \
--lr-scheduler-type linear \
--learning-rate 4e-5 \
--warmup-ratio 0.30 \
--per-device-train-batch-size 12 \
--per-device-eval-batch-size 12 \
--save-total-limit 1 \
--model-base  facebook/bart-base \
--run-id bart_base_model_crossed_context_enforcer_mod2 \
--fp16 \
--gradient-accumulation-steps 4 \
--output-dir trained_models_sum_boundary/  >> training_logs_bart_base_model_mod2_context_enforcer_crossed.out &