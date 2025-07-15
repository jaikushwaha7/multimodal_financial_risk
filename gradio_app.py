import gradio as gr
import torch
from config import config
from data.processor import TabularDataProcessor
from data.image_processor import ImageDataProcessor
from text.text_processor import TextDataProcessor
from models.encoders import TabularEncoder, TextEncoder
from models.attention import CrossModalAttention
import numpy as np
from PIL import Image

class DummyMultimodalModel(torch.nn.Module):
    def __init__(self, tabular_dim, text_dim=768, img_dim=(3, 224, 224), meta_dim=3, num_classes=3):
        super().__init__()
        self.tabular_encoder = TabularEncoder(tabular_dim)
        self.text_encoder = TextEncoder(text_dim)
        self.img_encoder = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(np.prod(img_dim), 64),
            torch.nn.ReLU(),
        )
        self.meta_encoder = torch.nn.Linear(meta_dim, 64)
        self.attn = CrossModalAttention(dim=64)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(64 * 4, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, num_classes)
        )

    def forward(self, tabular, text, image, metadata):
        t = self.tabular_encoder(tabular)
        x = self.text_encoder(text)
        i = self.img_encoder(image)
        m = self.meta_encoder(metadata)
        fusion = self.attn(t.unsqueeze(1), x.unsqueeze(1)).squeeze(1)
        combined = torch.cat([fusion, t, x, m], dim=1)
        return self.classifier(combined)

def predict(ticker):
    tab_proc = TabularDataProcessor()
    text_proc = TextDataProcessor()
    img_proc = ImageDataProcessor()

    tabular, metadata, _ = tab_proc.simulate_financials([ticker])
    tabular_scaled = tab_proc.preprocess(tabular)
    text = text_proc.batch_process([ticker])
    image = img_proc.batch_generate([ticker])

    model = DummyMultimodalModel(tabular.shape[1])
    model.eval()
    with torch.no_grad():
        out = model(
            torch.tensor(tabular_scaled, dtype=torch.float32),
            torch.tensor(text, dtype=torch.float32),
            image,
            torch.tensor(metadata, dtype=torch.float32)
        )
    pred = torch.argmax(out, dim=1).item()
    label_map = {0: "Safe", 1: "Medium", 2: "Risky"}

    explanation = f"""
### 🔍 Data Sources & Multimodal Fusion

- **Tabular**: Simulated financial metrics (e.g., P/E, ROE, D/E)
- **Text**: News articles fetched via NewsAPI, embedded using FinBERT
- **Image**: 1-year candlestick chart generated using yfinance & mplfinance
- **Metadata**: One-hot encoded sector information

All modalities are encoded independently and then **fused using attention** and concatenation.

### 🎯 Target Variable
Randomly assigned class for POC: `0 = Safe`, `1 = Medium`, `2 = Risky`

### 🧪 Sample Input
- Tabular: {tabular[0].tolist()}
- Text Embedding (shape): {text[0].shape}
- Metadata: {metadata[0].tolist()}
"""

    return f"Predicted Risk: {label_map[pred]}\n\n{explanation}"

tickers = config['sample_tickers']
iface = gr.Interface(
    fn=predict,
    inputs=gr.Dropdown(choices=tickers, label="Select Ticker"),
    outputs="markdown",
    title="🧠 Multimodal Risk Classifier",
    description="This demo uses simulated financials, chart images, and news text to classify stock risk."
)

if __name__ == "__main__":
    iface.launch()