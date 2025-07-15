# Multimodal Financial Risk Classifier (POC)
**This proof-of-concept project integrates multimodal data (financial metrics, news text, stock charts) to classify companies by risk levels using a hybrid AI model.**

**What It Does**
* Given a set of stock tickers, the pipeline:

* Simulates tabular data ( P/E, ROE)

* Fetches recent news articles and embeds them using FinBERT

*  Generates candlestick stock charts from yfinance and mplfinance

### Trains a model that fuses all modalities to predict a risk class (Safe, Medium, Risky)

## Folder Structure
```
multimodal_financial_risk/
│
├── main.py                      # Entry point: training loop, orchestration
├── config.py                   # Configs (paths, hyperparams, model sizes)
│
├── data/
│   ├── scraper.py              # FinancialDataScraper class
│   ├── processor.py            # FinancialDataProcessor class
│   └── image_processor.py      # ImageDataProcessor class
│
├── text/
│   └── text_processor.py       # TextDataProcessor class (FinBERT-based)
│
├── models/
│   ├── encoders.py             # TabularEncoder, TextEncoder, ImageEncoder
│   ├── attention.py            # CrossModalAttention
│   ├── multimodal_model.py     # Full MultimodalFinancialModel
│
├── train/
│   ├── dataset.py              # MultimodalFinancialDataset
│   └── trainer.py              # MultimodalFinancialTrainer
│
├── utils/
│   └── metrics.py              # Accuracy, F1, AUROC (optional)
│
├── data/raw/                   # Scraped CSVs
├── charts/                     # Saved images from chart generation
├── checkpoints/                # Saved model weights
└── README.md
```

Install the packages
```
uv pip install -r requirements.txt
```

Python functions:
* image_processor.py generating charts
* text_processor.py embedding financial news
* data/processor.py now includes TabularDataProcessor, which simulates tabular features and sector-based metadata for each ticker. This is ideal for prototyping and POC validation.
* main.py This script:
    - Loads tabular, text, image, and metadata features
    - Combines them in a DummyMultimodalModel
    - Trains using MultimodalFinancialTrainer
    - Operates on your selected 5 sample tickers


Sample gradio App output:
![alt text](/images/image.png)


## Explainer App (gradio_explainer_app.py)
- Gives overview of the data and how labeling is done
- Visulaization of the distribution
- Summary view wrt each sample Stocks(from the drop down)


Sample gradio App output:
![alt text](/images/image_2.pngage.png)

## Integration with LM Studio for integrating with LLMs

- Uses our test case help better understand the portfolio (User Friendly)
- Local run can be done for testing
- Used google/gemma3 (SLM) for the run and testing. (System was breathing file)

Few screenshots for representation:
![alt text](/images/image_llmapp1.png)
![alt text](/images/image_llmapp2.png)


