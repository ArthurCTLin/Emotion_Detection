import os 
import numpy as np 
import random
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def seed_everything(seed=20): 
    """Seed everything to ensure reproducibility"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True 


def get_latest_checkpoint(model_dir='./outputs/model'):
    if not os.path.exists(model_dir) or not any(d.startswith('checkpoint-') for d in os.listdir(model_dir)):
        raise FileNotFoundError(
            f"❌ Model is not found in `{model_dir}`。\nPlease execute `train.py` to train the model."
        )

    checkpoints = [d for d in os.listdir(model_dir) if d.startswith('checkpoint-')]
    return os.path.join(model_dir, checkpoints[0])



def compute_metrics(labels, logits):
    """
    Compute classification metrics: accuracy, f1, precision, recall, AUC, specificity.

    Args:
        y_true (np.ndarray): Ground truth labels (shape: [n_samples])
        y_logits (np.ndarray): Raw model logits (shape: [n_samples, n_classes])

    Returns:
        dict: Metric results
    """
    probs = F.softmax(torch.tensor(logits), dim=1).numpy()
    preds = np.argmax(probs, axis=1)

    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='micro')
    precision = precision_score(labels, preds, average='micro')
    recall = recall_score(labels, preds, average='micro')

    # Compute multiclass AUC using 'ovr' (One-vs-Rest), if possible
    try:
        if len(np.unique(labels)) > 2:
            auc = roc_auc_score(labels, probs, multi_class='ovr', average='weighted')
        else:
            auc = roc_auc_score(labels, probs[:, 1])
    except Exception as e:
        auc = 'N/A'

    # Specificity is tricky in multiclass — compute average specificity per class
    cm = confusion_matrix(labels, preds)
    specificity_per_class = []
    for i in range(len(cm)):
        tp = cm[i, i]
        fn = np.sum(cm[i, :]) - tp
        fp = np.sum(cm[:, i]) - tp
        tn = np.sum(cm) - (tp + fp + fn)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        specificity_per_class.append(specificity)

    avg_specificity = np.mean(specificity_per_class)

    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'auc': auc,
        'specificity': avg_specificity
    }

def compute_metrics_trainer(p):
    """
    Wrapper for Hugging Face Trainer.

    Args:
        p (transformers.EvalPrediction): Contains predictions and label_ids.

    Returns:
        dict: Metric results.
    """
    return compute_metrics(p.label_ids, p.predictions)

def plot_matrix(y_true, y_pred, label_map=None, save_path=None):
    
    """
    Plot the Confusion Matrix
    
    Args:
        y_true (list or array): True label 
        y_pred (list or array): Prediction
        label_map (dict, optional): The conversion of label number to text
        save_path (str, optional): path save the graph of confusion matrix
    """

    if label_map:
        labels = list(label_map.keys())
        display_labels = list(label_map.values())
    else:
        labels = None
        display_labels = None 

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    disp.plot()

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        #os.makedirs(os.path.dirname(save_path), exist_ok=True)
        #plt.savefig(f'{save_path}', dpi=300, bbox_inches='tight')
        plt.savefig(f'{save_path}/confusion_matrix.png', dpi=300, bbox_inches='tight')
        print(f"[Confusion Matrix] Saved to {save_path}")

    plt.show()

def plot_matrix_trainer(p, label_map=None, save_path=None):

    probs = F.softmax(torch.tensor(p.predictions), dim=1).numpy()
    preds = np.argmax(probs, axis=1)
    
    plot_matrix(p.label_ids, preds, label_map=label_map, save_path=save_path)
