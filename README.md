# 🧠 Emotion Classification Template (NLP)

This is a simple and extendable template for emotion classification using the [Hugging Face Emotion Dataset](https://huggingface.co/datasets/setfit/emotion). It serves as a base for building NLP classification models with Hugging Face Transformers.

## 📂 Dataset

We use the [SetFit/emotion](https://huggingface.co/datasets/setfit/emotion) dataset, which consists of six emotion classes:
* Classes: [sadness, joy, love, anger, fear, surprise]
* Statistics:
  * Training dataset: 16000
  * Validation dataset: 2000
  * Test dataset: 2000

## ⚙️ Installation

```bash
conda create -n emotion_detection python=3.8
conda activate emotion_detection

git clone https://github.com/ArthurCTLin/Emotion_Detection.git
cd Emotion_Detection

pip install -r requirements.txt
```

## 🚀 Usage

### 1️⃣ Train a model
```bash
python train.py
```

**Optional arguments:**

| Argument         | Description                         | Default                        |
|------------------|-------------------------------------|--------------------------------|
| `--model_name`   | Pretrained model name               | microsoft/deberta-v3-large     |
| `--num_classes`  | Number of classes                   | 6                              |
| `--max_length`   | max_length of tokenized input       | 512                            |
| `--weight_decay` | Strength of L2 regularization to prevent overfitting | 0.01          |
| `--warmup_ratio` | Warmup steps as a fraction of total training steps           | 0.0                             |
| `--epochs`       | Number of training epochs           | 10                             |
| `--batch_size`   | Training batch size                 | 8                              |
| `--output_dir`   | Output directory for model checkpoints | ./outputs/model             |

![image](https://github.com/user-attachments/assets/295aaf0e-09ec-4c6b-892d-8122822a7fcd)
| Accuracy | F1 score | Precision   | Sensitivity | Specificity | AUC |
|----------|----------|-------------|-------------|-------------|-----|
| 0.945    | 0.945    | 0.945       |0.945        | 0.988       |0.996|

### 2️⃣ Evaluate on test set
```bash
python evaluate.py
```

**Optional arguments:**

| Argument         | Description                         | Default                     |
|------------------|-------------------------------------|-----------------------------|
| `--model_path`   | Path to the trained checkpoint      | ./outputs/model             |
| `--output_dir`   | Directory to save evaluation results| ./outputs/test              |
| `--max_length`   | Maximum token sequence length       | 512                         |
| `--batch_size`   | Evaluation batch size               | 8                           |

![image](https://github.com/user-attachments/assets/db6d9c3c-569b-49fc-a8f7-65c072275bbc)
| Accuracy | F1 score | Precision   | Sensitivity | Specificity | AUC |
|----------|----------|-------------|-------------|-------------|-----|
| 0.934    | 0.934    | 0.934       |0.934        | 0.986       |0.997|

### 3️⃣ Predict a single example via command line
```bash
python inference.py --text "I feel amazing today!"
```

**Arguments:**

| Argument         | Description                         | Default             |
|------------------|-------------------------------------|---------------------|
| `--text`         | Text to analyze                     | (required)          |
| `--model_path`   | Path to model checkpoint            | ./outputs/model     |

### 4️⃣ Interactive Streamlit Demo
```bash
python demo.py
```
You can enter a sentence and get the predicted emotion interactively.
![image](https://github.com/user-attachments/assets/95dfd253-e9b2-4e6e-a16d-e25a4fa3e269)

### 5️⃣ Run on API
Start the API server:
```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```
`POST/predict`
```bash
"text": "I really do not want to work today!"
```
`Response:`
```bash
{
  "emotion": "sadness"
}
```

## 📌 Notes

- Make sure to run `train.py` first to train the model before using `evaluate.py`, `inference.py`, or `demo.py`.
- By default, the latest checkpoint under `./outputs/model` will be automatically detected.
- The trained model is **not uploaded to GitHub**. Please use `.gitignore` to avoid tracking large files.

## ✅ To Do

- [ ] 5-fold cross validation
- [ ] Docker 
