import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import math

FINBERT_MODEL = "ProsusAI/finbert"
finbert_tokenizer = None
finbert_model = None

def finbert_init():
    global finbert_tokenizer, finbert_model
    if finbert_tokenizer is None or finbert_model is None:
        return False
    try:
        finbert_tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL)
        finbert_model = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL)
        return True
    except Exception as e:
        print(f"Error initializing Finbert: {e}")
        return False
    
def finbert_score(text):

    if finbert_tokenizer is None or finbert_model is None:
        raise RuntimeError("Finbert model is not initialized. Call finbert_init().")
    inputs = finbert_tokenizer(text, return_tensors= 'pt', truncation=True, max_length=512)
    with torch.no_grad():
        outputs = finbert_model(**inputs)
        scores = softmax(outputs.logits.detach().cpu().numpy()[0])
        return float(scores[2] - scores[0]) 
