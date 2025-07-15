import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

class TabularDataProcessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.features = [
            'Market Cap', 'P/E', 'ROE', 'ROA', 'Debt/Equity',
            'Current Ratio', 'Operating Margin', 'Beta', 'Revenue Growth'
        ]
        self.metadata_fields = ['Sector_Tech', 'Sector_Finance', 'Sector_Other']

    def simulate_financials(self, tickers):
        np.random.seed(42)
        tabular = np.random.rand(len(tickers), len(self.features))
        metadata = np.zeros((len(tickers), len(self.metadata_fields)))
        for i in range(len(tickers)):
            metadata[i, i % len(self.metadata_fields)] = 1
        labels = np.random.randint(0, 3, len(tickers))
        return tabular, metadata, labels

    def preprocess(self, tabular):
        return self.scaler.fit_transform(tabular)
    
    def save_to_csv(self, tickers, tabular, metadata, labels, path="data/raw/financial_statements.csv"):
        import pandas as pd
        df = pd.DataFrame(tabular, columns=self.features)
        for i, name in enumerate(self.metadata_fields):
            df[name] = metadata[:, i]
        df['Ticker'] = tickers
        df['Label'] = labels
        df.to_csv(path, index=False)
