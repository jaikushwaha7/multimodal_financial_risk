import torch

config = {
    "data_path": "data/raw/financial_statements.csv",
    "image_dir": "charts/",
    "batch_size": 4,
    "epochs": 3,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "fusion_type": "hybrid",
    "hidden_dim": 64,
    "num_classes": 3,
    "text_model_name": "ProsusAI/finbert",
    "news_api_key": "e5939a6d51434a588c194c496dd95c41",
    "sample_tickers": ["AAPL", "TSLA", "GOOGL", "MSFT", "AMZN"]
}
