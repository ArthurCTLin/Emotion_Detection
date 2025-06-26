import numpy as np 
import pandas as pd 
import torch
from torch.utils.data import Dataset

def read_dataset(ds, mode):
    """
    Convert a HuggingFace Dataset split into a standardized pandas DataFrame

    Args:
        ds (DatasetDict): A dataset loaded via `load_dataset()` from HuggingFace.
        mode (str): Dataset split to use, e.g. 'train', 'validation', or 'test'.

    Returns:
        pd.DataFrame: DataFrame with columns ['Text', 'Label', 'Label_text']
    """
    texts = []
    labels = []
    label_texts = []
    for i in range(len(ds[mode])):
        texts.append(ds[mode][i]['text'])
        labels.append(ds[mode][i]['label'])
        label_texts.append(ds[mode][i]['label_text'])
    df = pd.DataFrame(list(zip(texts, labels, label_texts)),
                  columns=['Text', 'Label', 'Label_text'])
    return df 

def tokenized_text(text: str, tokenizer, max_length: int = 512, return_tensors=None):
    token = tokenizer(
        text,
        return_tensors=return_tensors,
        max_length=max_length,
        padding='max_length',
        truncation=True
    )
    return token

class CustomDataset:
    """
    A custom PyTorch Dataset class that converts a DataFrame of text data
    into tokenized input suitable for transformer models.

    Args:
        df (pd.DataFrame): DataFrame containing 'Text' and 'Label' columns.
        tokenizer (PreTrainedTokenizer): HuggingFace tokenizer instance.
        max_length (int): Maximum sequence length for tokenization.

    Returns:
        Dict[str, np.ndarray], int: A dictionary of tokenized inputs and an integer label.

    """
    def __init__(self, df, tokenizer, max_length):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        row = self.df.iloc[index]
        text = row.Text
        label = int(row.Label)
        token = tokenized_text(text, self.tokenizer, self.max_length)

        for k, v in token.items():
            token[k] = np.array(v, dtype=np.int64)
        return token, label


class SingleTextDataset(Dataset):
    """
    This class wraps a tokenized input (a dictionary of tensors) into a dataset
    with length 1, suitable for use with HuggingFace Trainer's `.predict()` method.

    Args:
        text (str): Input text to classify.
        tokenizer (PreTrainedTokenizerBase): Tokenizer to convert text to model inputs.
        max_length (int): Maximum sequence length for tokenization.
    Returns:
    |   Dict[str, torch.Tensor]: A dictionary of input tensors, suitable for model inference.
    """
    def __init__(self, text: str, tokenizer, max_length: int = 512):
        self.tokenized_input = tokenized_text(text, tokenizer, max_length, return_tensors='pt')

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return {key: val[0] for key, val in self.tokenized_input.items()}

def collate_fn(batch):
    """
    Collate function used to combine data and label into a dictionary

    Args:
        batch (List[Tuple[Dict[str, np.ndarray], int]]): Each item is a tuple of
            (tokenized inputs, label).

    Returns:
        Dict[str, torch.Tensor]: Batched input including input_ids, token_type_ids, attention_mask, and Label.
    """

    tokens, labels =  zip(*batch)
    keys = tokens[0].keys()
    data = {k: torch.from_numpy(np.stack([token[k] for token in tokens])) for k in keys}
    data['labels'] = torch.from_numpy(np.stack(labels))
    
    return data 
