import os
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
from torch import FloatTensor
from transformers import (
    T5ForConditionalGeneration,
    BartForConditionalGeneration,
    GPT2Model,
)
from dataclasses import dataclass
import torch
from typing import Optional, Union, Callable, Dict, List, Tuple
from transformers import (
    TrainingArguments,
    Trainer,
    PreTrainedModel,
    DataCollator,
    PreTrainedTokenizerBase,
    EvalPrediction,
    TrainerCallback,
)
import torch.nn as nn
from torch.utils.data import Dataset

from transformers.modeling_outputs import Seq2SeqLMOutput,BaseModelOutput,Seq2SeqModelOutput,BaseModelOutputWithPastAndCrossAttentions

@dataclass
class T5ModelOutput(BaseModelOutputWithPastAndCrossAttentions):
    context_boundary: Optional[Tuple[torch.LongTensor]] = None
    cleaned_mask: Optional[Union[FloatTensor, torch.LongTensor]] = None

@dataclass
class EncoderOutputs(BaseModelOutput):
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cleaned_mask: Optional[Union[FloatTensor, torch.LongTensor]] = None
    context_boundary: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class SentenceEmbeddingOutput(BaseModelOutput):
    token_embeddings: torch.FloatTensor = None
    sentence_embedding: torch.FloatTensor = None
    attention_mask: torch.LongTensor = None

@dataclass
class Seq2SeqModelOutputBoundary(Seq2SeqModelOutput):
    context_boundary: Optional[Tuple[torch.LongTensor]] = None
    
@dataclass
class Seq2SeqLMOutputBoundary(Seq2SeqLMOutput):
    context_boundary: Optional[Tuple[torch.LongTensor]] = None


@dataclass
class Transformers:
    model_base: str
    bart = BartForConditionalGeneration
    t5 = T5ForConditionalGeneration
    gpt = GPT2Model

    def resolve(self):
        if "bart" in self.model_base:
            return self.bart
        if "t5" in self.model_base:
            return self.t5
        if "gpt" in self.model_base:
            return self.gpt


def model_init(
        model_base,
        vocab_size,
        device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
):
    architecture = Transformers(model_base).resolve()
    generator = architecture.from_pretrained(model_base)  # type: ignore
    # update the tokens
    generator.resize_token_embeddings(vocab_size)  # type: ignore
    return generator.to(device)  # type: ignore


def get_training_arguments(
        output_dir,
        num_train_epochs,
        learning_rate,
        lr_scheduler_type,
        warmup_ratio,
        weight_decay,
        save_total_limit,
        save_strategy,
        evaluation_strategy,
        eval_steps,
        run_id,
        per_device_train_batch_size,
        verbose=False,
        gradient_accumulation_steps=1,
        fp16=True,
        **unused_args,
):
    return TrainingArguments(
        gradient_accumulation_steps=gradient_accumulation_steps,
        fp16=fp16,
        overwrite_output_dir=True,
        adafactor=False,
        load_best_model_at_end=True,
        output_dir=os.path.join(output_dir ,  run_id ),
        evaluation_strategy=evaluation_strategy,  # "epoch",
        save_strategy=save_strategy,  # 'epoch',
        lr_scheduler_type=lr_scheduler_type,
        learning_rate=learning_rate,
        save_total_limit=save_total_limit,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_train_batch_size,
        disable_tqdm=not verbose,
        eval_steps=eval_steps,
        save_steps=eval_steps,
    )


class CustomTrainer(Trainer):
    def __init__(
            self,
            device=None,
            model: Union[PreTrainedModel, nn.Module] = None,  # type: ignore
            args: TrainingArguments = None,  # type: ignore
            data_collator: Optional[DataCollator] = None,
            train_dataset: Optional[Dataset] = None,
            eval_dataset: Optional[Dataset] = None,
            tokenizer: Optional[PreTrainedTokenizerBase] = None,
            model_init: Callable[[], PreTrainedModel] = None,
            compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
            callbacks: Optional[List[TrainerCallback]] = None,
            optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (
                    None,
                    None,
            ),
            preprocess_logits_for_metrics: Callable[
                [torch.Tensor, torch.Tensor], torch.Tensor
            ] = None,  # type: ignore
            addition_input_keys=None,
    ):
        super().__init__(
            model,
            args,
            data_collator,
            train_dataset,
            eval_dataset,
            tokenizer,
            model_init,
            compute_metrics,
            callbacks,
            optimizers,
            preprocess_logits_for_metrics,
        )
        if addition_input_keys is None:
            addition_input_keys = []
        self.addition_input_keys = addition_input_keys

        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.device = device

    def compute_loss(self, model, batch, return_outputs=False):
        b_input_ids = batch["input_ids"].to(self.device)
        b_input_mask = batch["attention_mask"].to(self.device)
        b_labels = batch["labels"].to(self.device)
        decoder_attention_mask = batch["decoder_attention_mask"].to(self.device)

        additional_args = {key: batch.get(key, None) for key in self.addition_input_keys}

        outputs = model(
            b_input_ids,
            attention_mask=b_input_mask,
            decoder_attention_mask=decoder_attention_mask,
            labels=b_labels,
            **additional_args
        )
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        return (loss, outputs) if return_outputs else loss
