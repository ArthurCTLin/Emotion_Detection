from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification

def load_model(model_name_or_path: str, num_classes: int=None):
    """
    Create a new model for training or load a pretrained model for inference. 

    Args:
        model_name_or_path (str) Model name (e.g. "microsoft/deberta-v3-large") or path to a trained model directory.
        num_classes (int, optional): Number of output classes. If None, assumes loading pretrained model.

    Returns:
        tokenizer, model: Huggingface tokenizer and model instance 
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    if num_classes is not None:
        config = AutoConfig.from_pretrained(model_name_or_path, num_labels=num_classes)
        model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, config=config)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)

    return tokenizer, model
