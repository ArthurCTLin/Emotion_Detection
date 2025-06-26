import argparse
import torch
import numpy as np
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForSequenceClassification
from dataset import SingleTextDataset, tokenized_text
from model import load_model
from utils import get_latest_checkpoint

label_map = {
    0: 'sadness',
    1: 'joy',
    2: 'love',
    3: 'anger',
    4: 'fear',
    5: 'surprise'
}

def predict_emotion(text, tokenizer, model):
    dataset = SingleTextDataset(text, tokenizer)

    trainer = Trainer(
        model=model,
        args=TrainingArguments(output_dir='./outputs/inference', do_predict=True)
    )

    pred = trainer.predict(dataset)
    logits = torch.from_numpy(pred.predictions)
    probs = torch.softmax(logits, dim=1).cpu().detach().numpy()
    label_idx = int(np.argmax(probs, axis=1)[0])

    return label_map[label_idx]

def main():
    parser = argparse.ArgumentParser(description="Emotion Detection from Text")
    parser.add_argument('--text', type=str, required=True, help='Text to analyze')
    parser.add_argument('--model_dir', type=str, default='./outputs/model', help='Directory saved the trained model checkpoint')
    args = parser.parse_args()
    
    model_path = get_latest_checkpoint(args.model_dir)
    tokenizer, model = load_model(model_path)

    emotion = predict_emotion(args.text, tokenizer, model)
    print(f"\n🔍 Predicted Emotion: {emotion}")

if __name__ == '__main__':
    main()
