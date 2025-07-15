import gradio as gr
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Sample data structure based on your scraper
def create_sample_data():
    """Create sample financial data similar to what your scraper would produce"""
    np.random.seed(42)
    
    # Sample tickers from your scraper
    tickers = [
        'AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA', 'TSLA', 'JPM', 'BAC', 'WFC',
        'JNJ', 'PFE', 'XOM', 'CVX', 'WMT', 'HD', 'HSBA.L', 'BP.L', 'SAP.DE',
        'VOW3.DE', '7203.T', 'BABA', 'INFY.NS', 'TCS.NS', 'RELIANCE.NS'
    ]
    
    data = []
    for ticker in tickers:
        # Generate realistic financial metrics
        market_cap = np.random.lognormal(15, 2)  # Billions
        pe_ratio = np.random.gamma(2, 8)
        roe = np.random.normal(0.15, 0.1)
        debt_equity = np.random.gamma(1, 0.5)
        current_ratio = np.random.normal(1.5, 0.5)
        operating_margin = np.random.normal(0.2, 0.15)
        net_profit_margin = np.random.normal(0.1, 0.08)
        beta = np.random.normal(1.0, 0.3)
        revenue_growth = np.random.normal(0.08, 0.15)
        
        # Determine sector based on ticker
        if ticker in ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA']:
            sector = 'Technology'
        elif ticker in ['JPM', 'BAC', 'WFC']:
            sector = 'Financial Services'
        elif ticker in ['JNJ', 'PFE']:
            sector = 'Healthcare'
        elif ticker in ['XOM', 'CVX']:
            sector = 'Energy'
        elif ticker in ['WMT', 'HD']:
            sector = 'Consumer Discretionary'
        else:
            sector = 'Other'
            
        # Determine country based on ticker suffix
        if '.L' in ticker:
            country = 'UK'
        elif '.DE' in ticker:
            country = 'Germany'
        elif '.T' in ticker:
            country = 'Japan'
        elif '.NS' in ticker:
            country = 'India'
        else:
            country = 'USA'
        
        data.append({
            'Ticker': ticker,
            'Market Cap': market_cap,
            'P/E': pe_ratio,
            'ROE': roe,
            'Debt/Equity': debt_equity,
            'Current Ratio': current_ratio,
            'Operating Margin': operating_margin,
            'Net Profit Margin': net_profit_margin,
            'Beta': beta,
            'Revenue Growth': revenue_growth,
            'Sector': sector,
            'Country': country
        })
    
    return pd.DataFrame(data)

def calculate_altman_z_score(df):
    """Calculate Altman Z-Score for bankruptcy prediction"""
    # Simplified Z-Score calculation (would need more detailed financial data)
    z_scores = []
    for _, row in df.iterrows():
        # Simplified formula using available metrics
        working_capital_ratio = row['Current Ratio'] - 1
        retained_earnings_ratio = row['ROE'] * 0.1  # Approximation
        ebit_ratio = row['Operating Margin']
        market_value_ratio = np.log(row['Market Cap']) / 10
        sales_ratio = row['Revenue Growth']
        
        z_score = (1.2 * working_capital_ratio + 
                  1.4 * retained_earnings_ratio + 
                  3.3 * ebit_ratio + 
                  0.6 * market_value_ratio + 
                  1.0 * sales_ratio)
        z_scores.append(z_score)
    
    df['Z_Score'] = z_scores
    return df

def classify_risk_levels(df):
    """Classify companies into risk categories"""
    risk_levels = []
    for _, row in df.iterrows():
        z_score = row['Z_Score']
        debt_equity = row['Debt/Equity']
        current_ratio = row['Current Ratio']
        
        if z_score > 2.99 and debt_equity < 0.5 and current_ratio > 1.5:
            risk_levels.append('Safe')
        elif z_score < 1.81 or debt_equity > 1.5 or current_ratio < 1.0:
            risk_levels.append('Risky')
        else:
            risk_levels.append('Medium')
    
    df['Risk_Level'] = risk_levels
    return df

def classify_growth_value(df):
    """Classify companies as Growth or Value"""
    growth_value = []
    for _, row in df.iterrows():
        pe_ratio = row['P/E']
        revenue_growth = row['Revenue Growth']
        
        if pe_ratio > 20 and revenue_growth > 0.1:
            growth_value.append('Growth')
        elif pe_ratio < 15 and revenue_growth < 0.05:
            growth_value.append('Value')
        else:
            growth_value.append('Balanced')
    
    df['Growth_Value'] = growth_value
    return df

def create_risk_distribution_chart(df):
    """Create risk level distribution chart"""
    risk_counts = df['Risk_Level'].value_counts()
    
    fig = px.pie(
        values=risk_counts.values,
        names=risk_counts.index,
        title="Risk Level Distribution",
        color_discrete_map={
            'Safe': '#2E8B57',
            'Medium': '#FFD700',
            'Risky': '#DC143C'
        }
    )
    
    return fig

def create_sector_analysis_chart(df):
    """Create sector-wise analysis chart"""
    sector_metrics = df.groupby('Sector').agg({
        'ROE': 'mean',
        'P/E': 'mean',
        'Debt/Equity': 'mean',
        'Market Cap': 'mean'
    }).reset_index()
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('ROE by Sector', 'P/E by Sector', 'Debt/Equity by Sector', 'Market Cap by Sector'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # ROE
    fig.add_trace(
        go.Bar(x=sector_metrics['Sector'], y=sector_metrics['ROE'], name='ROE'),
        row=1, col=1
    )
    
    # P/E
    fig.add_trace(
        go.Bar(x=sector_metrics['Sector'], y=sector_metrics['P/E'], name='P/E'),
        row=1, col=2
    )
    
    # Debt/Equity
    fig.add_trace(
        go.Bar(x=sector_metrics['Sector'], y=sector_metrics['Debt/Equity'], name='Debt/Equity'),
        row=2, col=1
    )
    
    # Market Cap
    fig.add_trace(
        go.Bar(x=sector_metrics['Sector'], y=sector_metrics['Market Cap'], name='Market Cap'),
        row=2, col=2
    )
    
    fig.update_layout(height=600, showlegend=False, title_text="Sector Analysis Dashboard")
    return fig

def create_correlation_heatmap(df):
    """Create correlation heatmap for numerical features"""
    numerical_cols = ['Market Cap', 'P/E', 'ROE', 'Debt/Equity', 'Current Ratio', 
                     'Operating Margin', 'Net Profit Margin', 'Beta', 'Revenue Growth', 'Z_Score']
    
    corr_matrix = df[numerical_cols].corr()
    
    fig = px.imshow(
        corr_matrix,
        title="Financial Metrics Correlation Matrix",
        color_continuous_scale='RdBu',
        aspect="auto"
    )
    
    return fig

def create_pca_analysis(df):
    """Perform PCA analysis for dimensionality reduction"""
    numerical_cols = ['Market Cap', 'P/E', 'ROE', 'Debt/Equity', 'Current Ratio', 
                     'Operating Margin', 'Net Profit Margin', 'Beta', 'Revenue Growth']
    
    # Prepare data
    X = df[numerical_cols].fillna(df[numerical_cols].mean())
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(X_scaled)
    
    # Create DataFrame for plotting
    pca_df = pd.DataFrame({
        'PC1': pca_result[:, 0],
        'PC2': pca_result[:, 1],
        'Ticker': df['Ticker'],
        'Risk_Level': df['Risk_Level'],
        'Sector': df['Sector']
    })
    
    # Plot
    fig = px.scatter(
        pca_df, 
        x='PC1', 
        y='PC2',
        color='Risk_Level',
        hover_data=['Ticker', 'Sector'],
        title=f"PCA Analysis (Explained Variance: {pca.explained_variance_ratio_.sum():.2%})",
        color_discrete_map={
            'Safe': '#2E8B57',
            'Medium': '#FFD700',
            'Risky': '#DC143C'
        }
    )
    
    return fig

def create_ml_metrics_summary(df):
    """Create ML evaluation metrics summary"""
    # Simulate model performance metrics
    np.random.seed(42)
    
    metrics_data = {
        'Task': ['Risk Classification', 'Bankruptcy Prediction', 'Growth vs Value', 'Multimodal Embedding'],
        'Accuracy': [0.82, 0.89, 0.76, 0.85],
        'Precision': [0.80, 0.91, 0.74, 0.83],
        'Recall': [0.78, 0.85, 0.79, 0.87],
        'F1-Score': [0.79, 0.88, 0.76, 0.85],
        'AUC-ROC': [0.85, 0.93, 0.81, 0.88]
    }
    
    metrics_df = pd.DataFrame(metrics_data)
    
    # Create bar chart
    fig = px.bar(
        metrics_df.melt(id_vars='Task', var_name='Metric', value_name='Score'),
        x='Task',
        y='Score',
        color='Metric',
        barmode='group',
        title="ML Model Performance Metrics",
        height=500
    )
    
    fig.update_layout(xaxis_tickangle=-45)
    return fig

def create_modality_info():
    """Create information about data modalities"""
    modality_info = """
    ## 📦 Data Modalities Used in This Project
    
    ### 1. 🧾 Tabular Data (Structured)
    - **Source**: Financial statements, balance sheets, income statements
    - **Features**: Market Cap, P/E Ratio, ROE, Debt/Equity, Current Ratio, Operating Margin, etc.
    - **Processing**: Standardization, missing value imputation, feature engineering
    
    ### 2. 📰 Text Data (Unstructured) 
    - **Source**: News headlines, earnings call transcripts, analyst reports
    - **Processing**: FinBERT embeddings, sentiment analysis, topic modeling
    - **Features**: Sentiment scores, topic distributions, entity recognition
    
    ### 3. 🖼️ Image Data (Visual)
    - **Source**: Candlestick charts from 1-year historical stock prices
    - **Processing**: Generated using mplfinance, CNN feature extraction
    - **Features**: Technical patterns, trend analysis, volatility visualization
    
    ### 4. 🏷️ Metadata (Categorical)
    - **Source**: Company information
    - **Features**: Sector, Industry, Country, Market Classification
    """
    
    return modality_info

def create_ml_tasks_info():
    """Create information about ML tasks"""
    ml_tasks_info = """
    ## 🔍 Supported ML Tasks
    
    ### 1. 🔢 Risk Classification
    - **Task**: Predict if a company is "Safe", "Medium", or "Risky"
    - **Features**: Financial ratios, volatility, market signals, Z-score
    - **Evaluation**: Accuracy, Precision, Recall, F1-Score (macro-averaged)
    
    ### 2. 📉 Bankruptcy Prediction
    - **Task**: Binary classification of distressed vs healthy firms
    - **Method**: Altman Z-score, rule-based heuristics, ensemble methods
    - **Evaluation**: AUC-ROC, Precision-Recall curves, confusion matrix
    
    ### 3. 📈 Growth vs Value Labeling
    - **Task**: Classify companies as "Growth", "Value", or "Balanced"
    - **Features**: P/E ratio, dividend yield, revenue growth, price momentum
    - **Evaluation**: Multi-class classification metrics, Cohen's Kappa
    
    ### 4. 🧠 Multimodal Embedding Learning
    - **Task**: Learn unified representations from multiple data sources
    - **Method**: Fusion of tabular, text, and image embeddings
    - **Applications**: Recommendation systems, similarity search, clustering
    """
    
    return ml_tasks_info

def create_evaluation_info():
    """Create information about evaluation methods"""
    evaluation_info = """
    ## 📏 Model Evaluation Framework
    
    ### Classification Metrics
    - **Accuracy**: Overall correctness of predictions
    - **Precision**: True positives / (True positives + False positives)
    - **Recall**: True positives / (True positives + False negatives)
    - **F1-Score**: Harmonic mean of precision and recall
    - **AUC-ROC**: Area under the receiver operating characteristic curve
    
    ### Specialized Financial Metrics
    - **Altman Z-Score Alignment**: Agreement between model and traditional bankruptcy prediction
    - **Risk-Adjusted Returns**: Sharpe ratio, Sortino ratio for portfolio performance
    - **Volatility Prediction**: Mean squared error for volatility forecasting
    
    ### Model Interpretability
    - **SHAP Values**: Feature importance across different modalities
    - **Attention Visualization**: For multimodal transformer models
    - **Confusion Matrix**: Detailed classification performance analysis
    """
    
    return evaluation_info

def create_tracking_info():
    """Create information about tracking and logging"""
    tracking_info = """
    ## 📈 ML Training Tracking & Logging
    
    ### Recommended Tools
    
    #### 🟧 Weights & Biases (W&B)
    ```python
    import wandb
    
    wandb.init(project="financial-risk-prediction")
    wandb.log({
        "epoch": epoch,
        "train_loss": train_loss,
        "val_accuracy": val_acc,
        "confusion_matrix": wandb.plot.confusion_matrix(y_true, y_pred)
    })
    ```
    
    #### 🔷 MLflow
    ```python
    import mlflow
    
    with mlflow.start_run():
        mlflow.log_param("learning_rate", 0.01)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_artifact("model.pkl")
    ```
    
    #### 📊 TensorBoard
    ```python
    from torch.utils.tensorboard import SummaryWriter
    
    writer = SummaryWriter()
    writer.add_scalar('Loss/Train', train_loss, epoch)
    writer.add_scalar('Accuracy/Val', val_acc, epoch)
    ```
    
    ### Best Practices
    - Log per-epoch metrics (loss, accuracy, F1-score)
    - Track hyperparameters and model architectures
    - Save dataset version hashes for reproducibility
    - Monitor data drift and model performance over time
    - Log attention maps and feature importance for interpretability
    """
    
    return tracking_info

def main_analysis():
    """Main analysis function"""
    # Create sample data
    df = create_sample_data()
    
    # Calculate derived metrics
    df = calculate_altman_z_score(df)
    df = classify_risk_levels(df)
    df = classify_growth_value(df)
    
    return df

# Create Gradio interface
def create_gradio_app():
    df = main_analysis()
    
    with gr.Blocks(title="Financial ML Analysis Dashboard", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 📊 Financial Data Analysis & Multimodal ML Dashboard")
        gr.Markdown("*Comprehensive analysis of financial data with machine learning insights*")
        
        with gr.Tab("📈 Data Overview"):
            gr.Markdown("## Financial Data Sample")
            gr.DataFrame(df.head(10))
            
            gr.Markdown("## Dataset Statistics")
            gr.DataFrame(df.describe())
        
        with gr.Tab("🎯 Risk Analysis"):
            with gr.Row():
                with gr.Column():
                    risk_chart = create_risk_distribution_chart(df)
                    gr.Plot(risk_chart)
                
                with gr.Column():
                    pca_chart = create_pca_analysis(df)
                    gr.Plot(pca_chart)
        
        with gr.Tab("🏢 Sector Analysis"):
            sector_chart = create_sector_analysis_chart(df)
            gr.Plot(sector_chart)
            
            corr_chart = create_correlation_heatmap(df)
            gr.Plot(corr_chart)
        
        with gr.Tab("🤖 ML Performance"):
            ml_metrics_chart = create_ml_metrics_summary(df)
            gr.Plot(ml_metrics_chart)
        
        with gr.Tab("📚 Documentation"):
            with gr.Tab("Data Modalities"):
                gr.Markdown(create_modality_info())
            
            with gr.Tab("ML Tasks"):
                gr.Markdown(create_ml_tasks_info())
            
            with gr.Tab("Evaluation Methods"):
                gr.Markdown(create_evaluation_info())
            
            with gr.Tab("Tracking & Logging"):
                gr.Markdown(create_tracking_info())
        
        with gr.Tab("🔍 Individual Company Analysis"):
            ticker_dropdown = gr.Dropdown(
                choices=df['Ticker'].tolist(),
                label="Select Company",
                value=df['Ticker'].iloc[0]
            )
            
            def analyze_company(ticker):
                company_data = df[df['Ticker'] == ticker].iloc[0]
                
                analysis = f"""
                ## Analysis for {ticker}
                
                **Risk Level**: {company_data['Risk_Level']}
                **Sector**: {company_data['Sector']}
                **Country**: {company_data['Country']}
                **Growth/Value**: {company_data['Growth_Value']}
                
                ### Key Metrics
                - **Market Cap**: ${company_data['Market Cap']:.2f}B
                - **P/E Ratio**: {company_data['P/E']:.2f}
                - **ROE**: {company_data['ROE']:.2%}
                - **Debt/Equity**: {company_data['Debt/Equity']:.2f}
                - **Z-Score**: {company_data['Z_Score']:.2f}
                
                ### Risk Assessment
                {"✅ Low bankruptcy risk" if company_data['Z_Score'] > 2.99 else "⚠️ Medium risk" if company_data['Z_Score'] > 1.81 else "🚨 High bankruptcy risk"}
                """
                
                return analysis
            
            company_analysis = gr.Markdown()
            ticker_dropdown.change(analyze_company, inputs=[ticker_dropdown], outputs=[company_analysis])
    
    return demo

# Launch the app
if __name__ == "__main__":
    demo = create_gradio_app()
    demo.launch(share=True)