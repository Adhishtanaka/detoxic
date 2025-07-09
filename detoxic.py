import json
import re
import os
from fastapi import FastAPI
from pydantic import BaseModel
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from dotenv import load_dotenv
import yaml
from fastapi.middleware.cors import CORSMiddleware

model = load_model('toxic_comment_cnn_model.h5')


with open('tokenizer_config.json', 'r') as f:
    tokenizer_config = json.load(f)

tokenizer = Tokenizer(
    num_words=tokenizer_config['max_vocab_size'],
    oov_token=tokenizer_config['oov_token'],
    filters=tokenizer_config['filters'],
    lower=tokenizer_config['lower'],
    split=tokenizer_config['split'],
    char_level=tokenizer_config['char_level']
)
tokenizer.word_index = tokenizer_config['word_index']
MAX_SEQUENCE_LENGTH = tokenizer_config['max_sequence_length']

def preprocess_text(text):
    if text is None:
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = ' '.join(text.split())
    return text

app = FastAPI()

# Load environment variables
load_dotenv()
TRUSTED_URLS_PASSWORD = os.getenv('TRUSTED_URLS_PASSWORD', 'changeme')
TRUSTED_URLS_PATH = 'trusted_urls.yaml'

# Load trusted URLs from YAML
with open(TRUSTED_URLS_PATH, 'r') as f:
    trusted_config = yaml.safe_load(f)
    TRUSTED_URLS = trusted_config.get('trusted_urls', [])

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=TRUSTED_URLS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextIn(BaseModel):
    comment: str

class TextsIn(BaseModel):
    comments: list[str]

class TrustedUrlIn(BaseModel):
    url: str
    password: str

@app.post("/predict")
def predict(input: TextIn):
    text = preprocess_text(input.comment)
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
    prob = model.predict(padded)[0][0]
    return {
        "toxic_probability": float(prob),
        "is_toxic": bool(prob > 0.5)
    }

@app.post("/predict_batch")
def predict_batch(input: TextsIn):
    texts = [preprocess_text(t) for t in input.comments]
    seqs = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(seqs, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
    probs = model.predict(padded).flatten()
    return [
        {
            "toxic_probability": float(prob),
            "is_toxic": bool(prob > 0.5)
        }
        for prob in probs
    ]

@app.post("/add_trusted_url")
def add_trusted_url(input: TrustedUrlIn):
    if input.password != TRUSTED_URLS_PASSWORD:
        return {"success": False, "error": "Unauthorized"}
    if input.url in TRUSTED_URLS:
        return {"success": False, "error": "URL already in trusted list", "trusted_urls": TRUSTED_URLS}
    TRUSTED_URLS.append(input.url)
    with open(TRUSTED_URLS_PATH, 'w') as f:
        yaml.safe_dump({'trusted_urls': TRUSTED_URLS}, f)
    return {"success": True, "trusted_urls": TRUSTED_URLS}