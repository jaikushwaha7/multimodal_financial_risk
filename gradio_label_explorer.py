import gradio as gr
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from data.scraper import run_scraper

def explore_labels():
    try:
        df_z_and_rule = pd.read_csv("data/processed/financial_statements_with_z_and_rule.csv")
        df_price = pd.read_csv("data/processed/financial_with_price_labeled.csv")

        sns.set(style="whitegrid")
        fig, axs = plt.subplots(2, 2, figsize=(12, 5))

        sns.countplot(x="Price_Risk_Category", data=df_price, ax=axs[0, 0])
        axs[0, 0].set_title("📈 Price-Based Risk Category")

        sns.countplot(x="Z_Label", data=df_z_and_rule, ax=axs[0, 1])
        axs[0, 1].set_title("💼 Fundamental Z-Score Category")

        sns.histplot(df_z_and_rule["Altman_Z"], bins=30, kde=True, ax=axs[1, 0])
        axs[1, 0].set_title("💼 Fundamental Z-Score Distribution")

        sns.countplot(x="Rule_Label", data=df_z_and_rule, ax=axs[1, 1])
        axs[1, 1].set_title("📊 Rule-Based Risk Category")

        plt.tight_layout()
        if not os.path.exists("visuals"):
            os.makedirs("visuals")
        plot_path = "visuals/risk_category_distribution.png"
        plt.savefig(plot_path)
        plt.close()

        return (
            "### 🧾 Altman Z + Rule-Based Labels", df_z_and_rule.head(10).to_markdown(),
            "### 📈 Price Risk Categories", df_price.head(10).to_markdown(),
            plot_path
        )
    except Exception as e:
        return ("Error loading label files:", str(e), "")

def fetch_and_save(ticker_list):
    tickers = [t.strip().upper() for t in ticker_list.split(',') if t.strip()]
    run_scraper(tickers)
    return f"✅ Scraped and saved: {', '.join(tickers)}"

scraper_input = gr.Textbox(label="Enter Tickers (comma-separated)", placeholder="AAPL, TSLA, MSFT")

dashboard = gr.Interface(
    fn=explore_labels,
    inputs=[],
    outputs=["markdown", "markdown", "image"],
    title="📊 Label Inspection Dashboard",
    description="Explore rule-based, Altman Z-score, and price risk categories."
)

scraper_ui = gr.Interface(
    fn=fetch_and_save,
    inputs=scraper_input,
    outputs="text",
    title="📥 Financial Scraper",
    description="Run yfinance scraper and save financial data to data/raw."
)

tabs = gr.TabbedInterface(
    interface_list=[scraper_ui, dashboard],
    tab_names=["🔄 Run Scraper", "📊 Label Visualizer"]
)

if __name__ == "__main__":
    tabs.launch()
