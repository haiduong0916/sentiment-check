import torch
from transformers import RobertaForSequenceClassification, AutoTokenizer
import underthesea
import torch.nn.functional as F

MODEL_NAME = "wonrax/phobert-base-vietnamese-sentiment"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
model = RobertaForSequenceClassification.from_pretrained(MODEL_NAME)

def preprocess_text(text: str) -> str:
    text = text.strip().lower()
    tokens = underthesea.word_tokenize(text)
    return " ".join(tokens)

def classify_sentiment(text: str):
    processed = preprocess_text(text)
    input_ids = torch.tensor([tokenizer.encode(processed)])

    with torch.no_grad():
        outputs = model(input_ids)
        probs = F.softmax(outputs.logits, dim=-1).squeeze().tolist()

    labels = ["NEGATIVE 😞", "POSITIVE 😊", "NEUTRAL 😐"]
    pred_idx = int(torch.argmax(torch.tensor(probs)))

    return {
    "text": text,
    "sentiment": labels[pred_idx],
    "scores": {
        "NEGATIVE": round(probs[0], 3),
        "POSITIVE": round(probs[1], 3),
        "NEUTRAL": round(probs[2], 3)
    }
}

