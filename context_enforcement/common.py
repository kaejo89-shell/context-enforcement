import argparse


def create_training_args(parser=None):
    parser = argparse.ArgumentParser() if parser is None else parser
    parser.add_argument(
        "--model-base",
        "-mb",
        required=True,
        help="The type of transformer architecture",
    )
    parser.add_argument(
        "--output-dir",
        "-output-dir",
        default="trained_context_enforcers/",
        help="Location of where the trained models is saved",
    )
    parser.add_argument(
        "--task-type",
        required=False,
        help="Type of task: XSum, CNN-SUM",
    )
    parser.add_argument(
        "--run-id",
        "-run-id",
        type=str,
        default="",
        help="Id for the running"
    )
    parser.add_argument("--eval-steps", type=int, default=1000)
    parser.add_argument("--learning-rate", "-lr",
                        type=float, default=5e-5)

    parser.add_argument("--max-seq-len", default=512, type=int)
    parser.add_argument("--context-max-len", default=200, type=int)
    
    parser.add_argument('--evaluation-strategy', default='epoch')
    parser.add_argument('--save-strategy', default='epoch')

    parser.add_argument("--seed", default=10, type=int, help="Random seed")
    parser.add_argument("--lr-scheduler-type", default="cosine")
    parser.add_argument("--weight-decay",
                        type=float, default=0.0)
    parser.add_argument("--warmup-ratio",
                        required=False,
                        type=float, default=0.0)
    parser.add_argument("--num-train-epochs", type=int, default=10)

    parser.add_argument("--save-total-limit", type=int, default=1)
    parser.add_argument(
        "--per-device-train-batch-size",
        "-per-device-tbz",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--per-device-eval-batch-size",
        "-per-device-ebz",
        type=int,
        default=16,
    )
    # gradient_accumulation_steps
    parser.add_argument('--gradient-accumulation-steps',
                        type=int, default=1,
                        help="Gradient accumulation steps"
                        )
    parser.add_argument('--fp16', '-fp16', action="store_true")
    parser.add_argument("--verbose", "-verbose", action="store_true")
    # warmup_ratio save_total_limit per_device_eval_batch_size

    return parser
