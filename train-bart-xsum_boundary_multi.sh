export CUDA_VISIBLE_DEVICES=2
export PYTHONPATH=.:$PYTHONPATH
nohup python context_enforcement/trainers/train_bart.py \
--max-seq-len 800 \
--context-max-len 250 \
--context-max-len-list 150,450,250 \
--task-type xsum \
--num-train-epochs 10 \
--eval-steps 1000 \
--lr-scheduler-type linear \
--learning-rate 6e-5 \
--warmup-ratio 0.30 \
--per-device-train-batch-size 6 \
--per-device-eval-batch-size 6 \
--save-total-limit 1 \
--model-base  facebook/bart-base \
--run-id bart_base_model_context_enforcer_multi \
--fp16 \
--gradient-accumulation-steps 8 \
--output-dir trained_models_sum_boundary/  >> training_logs_bart_base_model_context_enforcer-multi.out &