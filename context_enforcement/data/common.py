import abc
import datetime
import functools
import os
import re
from dataclasses import field, dataclass
from typing import List, Dict, Optional, Union

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


def fill_blanks(sentence, tag, options):
    assert tag in sentence, f'Error {tag} not found in {sentence}'
    tag_options = {tag: options}
    extended1 = [
        functools.reduce(lambda a,
                                kv: a.replace(*kv),
                         tag_options.items(),
                         re.sub('\s+', ' ',
                                ss.strip().replace('\n', ' '))) for ss in [sentence]][0]
    return extended1


def read_sentences(file, lower=False) -> List[str]:
    with open(file, 'r', encoding="utf-8") as o_file:
        sentences = []
        for s in o_file.readlines():
            ss = s.strip().lower() if lower else s.strip()
            sentences.append(ss)
    return sentences


def write_to_file(content, filename):
    fil = filename + '.txt'
    if os.path.exists(fil):
        os.remove(fil)
    with open(fil, 'x') as fwrite:
        fwrite.writelines("%s\n" % s for s in content)
    print('Done')
    return


def round_to_n(n, p=1):
    dec, integ = np.modf(n)
    val = integ + np.round(dec, p)
    return val


def format_time(elapsed):
    return str(datetime.timedelta(seconds=int(round(elapsed))))


def normalize_whitespace(string):
    return re.sub(r'(\s)\1+', r'\1', string)


def create_text_tokenizer(model_base_name,
                          additional_tokens=None,
                          special_tokens=None,
                          ):
    """
    Creates a text tokenizer based on the specified models-base-name

    :param model_base_name:
    :param additional_tokens:
    :param special_tokens:
    :return:
    """
    if special_tokens is None:
        special_tokens = []
    if additional_tokens is None:
        additional_tokens = []
    tokenizer = AutoTokenizer.from_pretrained(model_base_name)
    tokenizer.add_tokens(additional_tokens)
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    return tokenizer


def pad_seq(
        seq: Union[np.ndarray, torch.Tensor, List], max_batch_len: int, pad_value: int
) -> List[int]:
    if len(seq) > max_batch_len:
        seq = seq.to(torch.long).unsqueeze(0)[:, :max_batch_len]
        return seq
    pads = torch.from_numpy(np.array([pad_value] * (max_batch_len - len(seq))))
    out = torch.concat([seq, pads], -1).to(torch.long).unsqueeze(0)
    return out


@dataclass
class Features:
    input_ids: Union[np.ndarray, torch.Tensor]
    attention_mask: Union[np.ndarray, torch.Tensor]
    labels: Optional[List[int]] = field(default_factory=list)
    decoder_attention_mask: Optional[List[int]] = field(default_factory=list)


class SmartCollator():
    def __init__(self,
                 pad_token_id: int,
                 context_max_len: int,
                 context_sampling_bounds: tuple,
                 label_pad_token_id: int = -100,
                 max_len: int = 512,
                 is_inference: bool = False):
        super().__init__()
        self.pad_token_id = pad_token_id
        self.label_pad_token_id = label_pad_token_id
        self.max_len = max_len
        self.is_inference = is_inference
        self.context_max_len = context_max_len
        self.context_sampling_bounds = context_sampling_bounds

    def __call__(self, batch: List[Features]) -> Dict[str, torch.Tensor]:
        batch_inputs: List = list()
        batch_attention_masks: List = list()
        decoder_attention_mask: List = list()
        labels: List = list()
        max_size = min([max([len(ex.input_ids)
                             for ex in batch]), self.max_len])

        max_size_output = min(
            [max([len(ex.labels) for ex in batch]), self.max_len])  # type: ignore

        for item in batch:
            batch_inputs += [pad_seq(item.input_ids,
                                     max_size, self.pad_token_id)]
            batch_attention_masks += [
                pad_seq(item.attention_mask, max_size, 0)]

            if not self.is_inference:
                labels += [pad_seq(item.labels, max_size_output,
                                   self.label_pad_token_id)]
                decoder_attention_mask += [
                    pad_seq(item.decoder_attention_mask, max_size_output, 0)
                ]

        input_ids = torch.concat(batch_inputs, 0)
        attention_mask = torch.concat(batch_attention_masks, 0)
        labels = torch.concat(labels, 0)
        decoder_attention_mask = torch.concat(decoder_attention_mask, 0)

        # Compute the context bounds for this batch
        boundary = np.random.uniform(self.context_sampling_bounds[0], self.context_sampling_bounds[1])
        boundary_start = int(input_ids.shape[-1] * boundary)
        boundary_end = boundary_start + self.context_max_len

        if not self.is_inference:
            return dict(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                decoder_attention_mask=decoder_attention_mask,
                boundary=(boundary_start, boundary_end)
            )
        else:
            return dict(
                input_ids=torch.concat(batch_inputs, 0),
                attention_mask=torch.concat(batch_attention_masks, 0),
                boundary=(boundary_start, boundary_end)
            )


class DatasetProcessor(Dataset, metaclass=abc.ABCMeta):
    """
    Class handles the creation pytorch dataset object.

    """

    def __init__(self, tokenizer, data, use_special_token=True):
        self.tokenizer = tokenizer
        self.data = data
        self.use_special_token = use_special_token

    def __len__(self, ):
        return len(self.data)

    def __getitem__(self, idx):
        return self._process_data(self.data[idx])

    @abc.abstractmethod
    def _process_data(self, param):
        raise NotImplemented("Function not implemented")
