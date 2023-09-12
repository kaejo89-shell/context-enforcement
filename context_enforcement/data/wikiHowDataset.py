from datasets import load_dataset

from context_enforcement.data.common import Features, DatasetProcessor,normalize_whitespace


def load_raw_wikihow(data_path):
    """
    Gets the Wikihow dataset from huggingface

    :return:
    """
    dataset = load_dataset("wikihow", "all", data_path)
    return dataset


class WikiHowDataset(DatasetProcessor):
    def __init__(self, tokenizer, data, use_special_token=True):
        super().__init__(
            tokenizer=tokenizer, data=data, use_special_token=use_special_token
        )

    def _process_data(self, data_point):
        document = data_point['text']
        document = normalize_whitespace(document.replace('.\n','. ').replace('\n','').strip())

        summary = data_point.get('headline',None)
        if summary is not None:
            summary = normalize_whitespace(
                data_point.get('headline', None).replace('.\n', '. ').replace('\n', '').strip())

        passage_pack = self.tokenizer(
            document,
            add_special_tokens=self.use_special_token,
            return_tensors="pt",
        )

        passage_seq = passage_pack["input_ids"].flatten()
        passage_attention = passage_pack["attention_mask"].flatten()

        features = Features(input_ids=passage_seq, attention_mask=passage_attention)

        if summary is not None:
            label_pack = self.tokenizer(
                summary, return_tensors="pt", add_special_tokens=self.use_special_token
            )
            label_seq = label_pack["input_ids"].flatten()
            label_attention = label_pack["attention_mask"].flatten()
            features.decoder_attention_mask = label_attention
            features.labels = label_seq
        return features


def create_wikihow_dataset(tokenizer,wikihow_data_path):
    dataset = load_raw_wikihow(data_path=wikihow_data_path)
    train_data = WikiHowDataset(
        tokenizer,
        dataset["train"],
    )
    test_data = WikiHowDataset(
        tokenizer,
        dataset["test"],
    )
    val_data = WikiHowDataset(
        tokenizer,
        dataset["validation"],
    )

    return {"train": train_data, "validation": val_data, "test": test_data}
