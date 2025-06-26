from transformers import Trainer
from model import load_model
from dataset import read_dataset, CustomDataset, collate_fn
from utils import compute_metrics_trainer, plot_matrix_trainer, get_latest_checkpoint
from datasets import load_dataset
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default='./outputs/model')
    parser.add_argument('--output_dir', type=str, default='./outputs/test')
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=8)
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Load test set
    raw_ds = load_dataset('SetFit/emotion')
    test_df = read_dataset(raw_ds, 'test')
    label_map = dict(test_df[['Label', 'Label_text']].drop_duplicates().sort_values(by='Label').values)

    # Load model 
    model_path = get_latest_checkpoint(args.model_dir)
    tokenizer, model = load_model(model_path)
    test_ds = CustomDataset(test_df, tokenizer, args.max_length)

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=collate_fn
    )

    # Predict
    test_output = trainer.predict(test_ds)
    scores = compute_metrics_trainer(test_output)
    plot_matrix_trainer(test_output, label_map, args.output_dir)

    print("Test metrics:")
    for k, v in scores.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()
