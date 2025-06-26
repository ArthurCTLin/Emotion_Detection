import streamlit as st
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from dataset import SingleTextDataset
from utils import get_latest_checkpoint

# Emotion label mapping
label_map = {
    0: 'sadness',
    1: 'joy',
    2: 'love',
    3: 'anger',
    4: 'fear',
    5: 'surprise'
}

@st.cache_resource
def load_model(model_dir='./outputs/model'):
    checkpoint_path = get_latest_checkpoint(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)
    return tokenizer, model

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

# ─────────────────────────────── UI ─────────────────────────────── #
st.title("🎭 Emotion Recognition")
user_input = st.text_area("Please enter a piece of text to analyze its emotion:", height=300)

if st.button("Analyze Emotion..."):
    if user_input.strip():
        try:
            tokenizer, model = load_model()
            emotion = predict_emotion(user_input, tokenizer, model)
            st.success(f"🔍 Predicted Emotion: **{emotion}**")
        except FileNotFoundError as e:
            st.error(str(e))
    else:
        st.warning("⚠️ Please enter some text.")
