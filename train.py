import argparse
import os
from transformers import TrainingArguments, Trainer
from datasets import load_dataset

from dataset import read_dataset, CustomDataset, collate_fn
from model import load_model
from utils import compute_metrics_trainer, plot_matrix_trainer, seed_everything


def parse_args():
    parser = argparse.ArgumentParser(description="Train an emotion detection model")

    # Basic settings
    parser.add_argument('--seed', type=int, default=42)

    # Model and training setting
    parser.add_argument('--model_name', type=str, default='microsoft/deberta-v3-large')
    parser.add_argument('--num_classes', type=int, default=6)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_ratio', type=float, default=0.0)
    parser.add_argument('--lr', type=float, default=1e-5)

    # Scheduler & optimizer
    parser.add_argument('--lr_schedule_type', type=str, default='cosine')
    parser.add_argument('--optimizer', type=str, default='adamw_torch')

    # Mixed precision
    # parser.add_argument('--fp16', action='store_true', help='Enable FP16 training')

    # Output
    parser.add_argument('--output_dir', type=str, default='./outputs')

    return parser.parse_args()


def main():
    args = parse_args()

    # Seed everything
    seed_everything(args.seed)

    # Load dataset
    raw_ds = load_dataset('SetFit/emotion')
    train_df = read_dataset(raw_ds, 'train')
    valid_df = read_dataset(raw_ds, 'validation')

    # Load model and tokenizer
    tokenizer, model = load_model(args.model_name, args.num_classes)

    # Dataset preprocessing
    train_ds = CustomDataset(train_df, tokenizer, args.max_length)
    valid_ds = CustomDataset(valid_df, tokenizer, args.max_length)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=f'{args.output_dir}/model',
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        optim=args.optimizer,
        lr_scheduler_type=args.lr_scheduler_type,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        logging_first_step=True,
        report_to='none',
        fp16=True
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        data_collator=collate_fn,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_trainer,
    )

    # Train!
    trainer.train()

    # Label Map
    emotion_map = dict(
        valid_df[['Label', 'Label_text']].drop_duplicates().sort_values(by='Label').values
    )
    
    # Evaluation
    valid_output = trainer.predict(valid_ds)
    plot_matrix_trainer(valid_output, emotion_map, f'{args.output_dir}/valid')
    print('The evaluation metrics:')
    scores = compute_metrics_trainer(valid_output)
    for key, value in scores.items():
        print(key, ': ',  value)

if __name__ == '__main__':
    main()
