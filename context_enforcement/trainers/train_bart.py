import os
from typing import List

os.environ["WANDB_DISABLED"] = "true"
from context_enforcement.data.common import create_text_tokenizer, SmartCollator
from context_enforcement.models.bart_context_enforcer_2 import (
    BartForContextualRecovery,
    BartForContextualRecoveryMultiLoss,
)
from context_enforcement.models.common import CustomTrainer, get_training_arguments
from context_enforcement.trainers.common import (
    add_context_enforcement_args,
    get_dataset_specified_tasks,
    create_training_args,
)
import ast
import nltk
from transformers import BartConfig, BartForConditionalGeneration
import torch
from transformers.trainer_callback import EarlyStoppingCallback
import pickle as pk
from pytorch_lightning import seed_everything

seed_everything(1376)

nltk.download("punkt")


def model_init(
    vocab_size,
    model_base="facebook/bart-base",
    context_num_heads=1,
    context_max_len=200,
    context_max_len_list: List = [200],
    context_sampling_bounds=(0.1, 0.45),
    device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
    is_baseline=False,
):
    def build_model():
        bart_config = BartConfig.from_pretrained(model_base)

        if not is_baseline:
            bart_config.context_num_heads = context_num_heads
            bart_config.context_max_len = context_max_len
            bart_config.context_sampling_bounds = context_sampling_bounds
            bart_config.context_max_len_list = context_max_len_list

        if is_baseline:
            model_class_name = BartForConditionalGeneration
        else:
            if context_max_len_list is not None and len(context_max_len_list) > 1:
                model_class_name = BartForContextualRecoveryMultiLoss
            else:
                model_class_name = BartForContextualRecovery

        generator = model_class_name.from_pretrained(
            model_base,
            config=bart_config,
        )

        # update the tokens
        generator.resize_token_embeddings(vocab_size)  # type: ignore
        return generator.to(device)  # type: ignore

    return build_model


if __name__ == "__main__":
    parser = create_training_args()
    parser = add_context_enforcement_args(parser)

    arguments = parser.parse_args()
    configs = vars(arguments)

    tokenizer = create_text_tokenizer(configs["model_base"])
    is_baseline = arguments.is_baseline
    context_max_len = configs.get("context_max_len", 250)
    context_max_len_list = configs.get("context_max_len_list")
    context_max_len_list = (
        ast.literal_eval(context_max_len_list[0])
        if context_max_len_list
        else [context_max_len_list]
    )

    if type(context_max_len_list) is not list:
        context_max_len_list = list(context_max_len_list)

    context_sampling_bounds = configs.get("context_sampling_bounds", (0.15, 0.45))

    task_dataset_gen = get_dataset_specified_tasks(configs["task_type"])

    train_dataset = None
    eval_dataset = None
    test_dataset = None
    if task_dataset_gen is not None:
        raw_dataset = task_dataset_gen(
            tokenizer=tokenizer,
        )
        train_dataset = raw_dataset["train"]
        eval_dataset = raw_dataset["validation"]
        test_dataset = raw_dataset["test"]

    model_builder = model_init(
        vocab_size=len(train_dataset.tokenizer),
        model_base=arguments.model_base,
        is_baseline=is_baseline,
        context_max_len=context_max_len,
        context_sampling_bounds=context_sampling_bounds,
        context_max_len_list=context_max_len_list,
    )

    training_arguments = get_training_arguments(**configs)
    custom_trainer = CustomTrainer(
        model=model_builder(),
        args=training_arguments,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=SmartCollator(
            pad_token_id=train_dataset.tokenizer.pad_token_id,
            context_max_len=context_max_len,
            context_sampling_bounds=context_sampling_bounds,
            max_len=arguments.max_seq_len,
        ),  # type: ignore
        callbacks=[EarlyStoppingCallback(early_stopping_patience=20)],
        addition_input_keys=["context_boundary"] if not is_baseline else [],
    )

    custom_trainer.train()

    output_path = os.path.join(arguments.output_dir, arguments.run_id, "train_args.ap")
    pk.dump(
        arguments,
        open(output_path, "wb"),
    )
