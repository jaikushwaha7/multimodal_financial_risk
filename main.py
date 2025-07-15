import torch
from torch.utils.data import DataLoader
from config import config
from data.processor import TabularDataProcessor
from data.image_processor import ImageDataProcessor
from text.text_processor import TextDataProcessor
from train.dataset import MultimodalFinancialDataset
from train.trainer import MultimodalFinancialTrainer
from models.encoders import TabularEncoder, TextEncoder
from models.attention import CrossModalAttention
import numpy as np
import wandb

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

if __name__ == '__main__':
    wandb.init(project="multimodal-financial-risk", config=config)
    tickers = config['sample_tickers']

    # Process all modalities
    tabular_processor = TabularDataProcessor()
    image_processor = ImageDataProcessor()
    text_processor = TextDataProcessor()

    tabular, metadata, labels = tabular_processor.simulate_financials(tickers)
    tabular = tabular_processor.preprocess(tabular)
    text = text_processor.batch_process(tickers)
    images = image_processor.batch_generate(tickers)

    dataset = MultimodalFinancialDataset({
        'tabular': tabular,
        'text': text,
        'images': images,
        'metadata': metadata,
        'labels': labels
    })
    loader = DataLoader(dataset, batch_size=config['batch_size'])

    model = DummyMultimodalModel(tabular.shape[1])
    trainer = MultimodalFinancialTrainer(model, config['device'])

    for epoch in range(config['epochs']):
        loss, acc = trainer.train_epoch(loader)
        wandb.log({"epoch": epoch+1, "loss": loss, "accuracy": acc})
        print(f"Epoch {epoch+1}: Loss={loss:.4f}, Acc={acc:.4f}")
