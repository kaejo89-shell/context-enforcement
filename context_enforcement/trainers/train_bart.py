import os

from context_enforcement.common import create_training_args

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["WANDB_DISABLED"] = "true"
from context_enforcement.data.common import create_text_tokenizer, SmartCollator
from context_enforcement.models.bart_context_enforcer import BartForContextualRecovery
from context_enforcement.models.common import CustomTrainer, get_training_arguments
from context_enforcement.trainers.common import get_dataset_specified_tasks

import nltk
from transformers import BartConfig
import torch
from transformers.trainer_callback import EarlyStoppingCallback
import pickle as pk

nltk.download("punkt")


def model_init(
        vocab_size,
        model_base="facebook/bart-base",
        context_num_heads=1,
        device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
):
    def build_model():
        bart_config = BartConfig.from_pretrained(model_base)
        bart_config.context_num_heads = context_num_heads
        generator = BartForContextualRecovery.from_pretrained(
            model_base,
            config=bart_config,
        )

        # update the tokens
        generator.resize_token_embeddings(vocab_size)  # type: ignore
        return generator.to(device)  # type: ignore

    return build_model


if __name__ == "__main__":
    parser = create_training_args()
    arguments = parser.parse_args()
    configs = vars(arguments)

    tokenizer = create_text_tokenizer(configs["model_base"])

    task_dataset_gen = get_dataset_specified_tasks(configs["task_type"])

    train_dataset = None
    eval_dataset = None
    test_dataset = None
    if task_dataset_gen is not None:
        raw_dataset = task_dataset_gen(tokenizer=tokenizer, )
        train_dataset = raw_dataset['train']
        eval_dataset = raw_dataset['validation']
        test_dataset = raw_dataset['test']

    model_builder = model_init(
        vocab_size=len(train_dataset.tokenizer),
        model_base=arguments.model_base,
    )

    training_arguments = get_training_arguments(**configs)
    custom_trainer = CustomTrainer(
        model=model_builder(),
        args=training_arguments,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=SmartCollator(
            pad_token_id=train_dataset.tokenizer.pad_token_id,
            context_max_len=configs.get("context_max_len", 100),
            context_sampling_bounds=configs.get("context_sampling_bounds",
                                                (0.1, 0.45)),
            max_len=arguments.max_seq_len,
        ),  # type: ignore
        callbacks=[EarlyStoppingCallback(early_stopping_patience=20)],
    )

    custom_trainer.train()

    output_path = os.path.join(arguments.output_dir, arguments.run_id, "train_args.ap")
    pk.dump(
        arguments,
        open(output_path, "wb"),
    )
