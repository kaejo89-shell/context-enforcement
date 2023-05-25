from context_enforcement.data.XsumDataset import create_xsum_dataset


def get_dataset_specified_tasks(task_type=None):
    if task_type is None:
        return None

    if task_type.lower() in ['xsum', 'xsummarization']:
        return create_xsum_dataset
