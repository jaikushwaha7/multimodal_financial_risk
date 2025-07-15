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
├── main.py                      # Main training and orchestration script
├── config.py                    # Configuration (paths, hyperparams, model names, etc.)
├── compute_zscore.py            # Altman Z-score calculation and labeling
├── rule_labeling.py             # Rule-based and price risk labeling
├── setup.py                     # Python package setup
├── requirements.txt             # Python dependencies
├── README.md                    # Project documentation
├── Readme_Labelling.md          # Labeling dashboard documentation
├── .gitignore                   # Git ignore rules
│
├── data/
│   ├── scraper.py               # Financial data scraping (yfinance)
│   ├── processor.py             # Tabular data simulation and preprocessing
│   ├── image_processor.py       # Chart image generation and processing
│   └── __init__.py
│   └── raw/                     # Scraped CSVs
│
├── text/
│   ├── text_processor.py        # News fetching and embedding (FinBERT, NewsAPI)
│   └── __init__.py
│
├── models/
│   ├── encoders.py              # Tabular, text, image encoders
│   ├── attention.py             # Cross-modal attention
│   ├── multimodal_model.py      # Full multimodal model
│   └── __init__.py
│
├── train/
│   ├── dataset.py               # Multimodal dataset class
│   ├── trainer.py               # Training loop and logic
│   └── __init__.py
│
├── utils/
│   ├── metrics.py               # Accuracy and metrics functions
│   ├── create_structure.py      # Script to create folder structure
│   ├── readme_FolderStructure.md
│   └── __init__.py
│
├── charts/                      # Saved chart images
├── checkpoints/                 # Saved model weights
├── images/                      # App screenshots and chart images
├── wandb/                       # Weights & Biases logs
├── notebook_testing.ipynb       # Jupyter notebook for testing
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


