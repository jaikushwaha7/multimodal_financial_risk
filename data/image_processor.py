import os
import yfinance as yf
import mplfinance as mpf
from PIL import Image
import torch
import torchvision.transforms as transforms

class ImageDataProcessor:
    def __init__(self, save_path="charts/"):
        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def generate_chart(self, ticker: str) -> torch.Tensor:
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="1y")

            if hist.empty:
                print(f"⚠️ No chart data for {ticker}")
                return torch.zeros(3, 224, 224)

            chart_file = os.path.join(self.save_path, f"{ticker}_chart.png")
            mpf.plot(hist, type='candle', mav=(20, 50), volume=True,
                     savefig=chart_file)

            img = Image.open(chart_file).convert("RGB")
            return self.transform(img)

        except Exception as e:
            print(f"❌ Chart generation failed for {ticker}: {e}")
            return torch.zeros(3, 224, 224)

    def batch_generate(self, tickers: list) -> torch.Tensor:
        return torch.stack([self.generate_chart(t) for t in tickers])
