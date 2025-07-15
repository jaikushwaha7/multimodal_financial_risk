import torch
import torch.nn as nn


class MultimodalFinancialTrainer:
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch in dataloader:
            tabular = batch['tabular'].to(self.device)
            text = batch['text'].to(self.device)
            image = batch['image'].to(self.device)
            metadata = batch['metadata'].to(self.device)
            labels = batch['label'].to(self.device).view(-1)

            # Debug print
            print(f"Batch shapes - Tabular: {tabular.shape}, "
                  f"Text: {text.shape}, "
                  f"Images: {image.shape}, "
                  f"Labels: {labels.shape}")

            outputs = self.model(tabular, text, image, metadata )
            loss = self.criterion(outputs, labels)

            self.optimizer.zero_grad()
            outputs = self.model(tabular, text, image, metadata)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        return total_loss / len(dataloader), correct / total

    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in dataloader:
                tabular = batch['tabular'].to(self.device)
                text = batch['text'].to(self.device)
                image = batch['image'].to(self.device)
                metadata = batch['metadata'].to(self.device)
                labels = batch['label'].to(self.device).view(-1)

                outputs = self.model(tabular, text, image, metadata)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return total_loss / len(dataloader), correct / total