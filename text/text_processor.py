import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from newsapi import NewsApiClient
from config import config


class TextDataProcessor:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(config["text_model_name"])
        self.model = AutoModel.from_pretrained(config["text_model_name"])
        self.newsapi = NewsApiClient(api_key=config["news_api_key"])
        self.model.eval()

    def fetch_articles(self, ticker):
        try:
            company = ticker if "." not in ticker else ticker.split(".")[0]
            articles = self.newsapi.get_everything(
                q=company,
                language='en',
                sort_by='relevancy',
                page_size=5
            )
            texts = [a['content'] or "" for a in articles['articles']]
            return " ".join(texts)
        except Exception as e:
            print(f"⚠️ Failed to fetch news for {ticker}: {e}")
            return "No news available."

    def encode_text(self, text):
        try:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True,
                                    padding=True, max_length=512)
            with torch.no_grad():
                outputs = self.model(**inputs)
            return outputs.last_hidden_state[:, 0, :].squeeze().numpy()
        except Exception as e:
            print(f"❌ Text encoding failed: {e}")
            return np.zeros(768)

    def process(self, ticker):
        raw_text = self.fetch_articles(ticker)
        return self.encode_text(raw_text)

    def batch_process(self, tickers):
        return np.array([self.process(t) for t in tickers])
