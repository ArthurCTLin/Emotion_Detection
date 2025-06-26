from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from inference import predict_emotion
from utils import get_latest_checkpoint
from model import load_model

app = FastAPI()

label_map = {
    0: 'sadness',
    1: 'joy',
    2: 'love',
    3: 'anger',
    4: 'fear',
    5: 'surprise'
}

model_path = get_latest_checkpoint('./outputs/model')
tokenizer, model = load_model(model_path)

class TextInput(BaseModel):
    text: str

@app.post("/predict")
def predict(input: TextInput):
    if not input.text.strip():
        raise HTTPException(status_code=400, detail="Text is empty")
    emotion = predict_emotion(input.text, tokenizer, model)
    return {"emotion": emotion}
