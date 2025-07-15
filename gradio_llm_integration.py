import gradio as gr
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import requests
from datetime import datetime
import asyncio
import aiohttp
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

# LM Studio Integration Class
class LMStudioClient:
    def __init__(self, base_url: str = "http://localhost:1234"):
        self.base_url = base_url
        self.api_endpoint = f"{base_url}/v1/chat/completions"
        
    async def generate_response(self, messages: List[Dict], 
                              model: str = "local-model",
                              temperature: float = 0.7,
                              max_tokens: int = 2000) -> str:
        """Generate response using LM Studio API"""
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.api_endpoint, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result['choices'][0]['message']['content']
                    else:
                        return f"Error: {response.status} - {await response.text()}"
        except Exception as e:
            return f"Connection error: {str(e)}"
    
    def sync_generate_response(self, messages: List[Dict], **kwargs) -> str:
        """Synchronous wrapper for generate_response"""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(self.generate_response(messages, **kwargs))

# Enhanced Financial Analysis with LLM Integration
class LLMFinancialAnalyzer:
    def __init__(self, llm_client: LMStudioClient):
        self.llm_client = llm_client
        
    def create_financial_context(self, df: pd.DataFrame) -> str:
        """Create context string from financial data"""
        context = f"""
        Financial Dataset Summary:
        - Total companies: {len(df)}
        - Sectors: {', '.join(df['Sector'].unique())}
        - Countries: {', '.join(df['Country'].unique())}
        - Risk levels: {df['Risk_Level'].value_counts().to_dict()}
        - Average P/E: {df['P/E'].mean():.2f}
        - Average ROE: {df['ROE'].mean():.2%}
        - Average Debt/Equity: {df['Debt/Equity'].mean():.2f}
        """
        return context
    
    def analyze_company_with_llm(self, ticker: str, company_data: pd.Series) -> str:
        """Generate comprehensive company analysis using LLM"""
        messages = [
            {
                "role": "system",
                "content": """You are a senior financial analyst with expertise in fundamental analysis, 
                risk assessment, and investment recommendations. Provide detailed, actionable insights 
                based on financial metrics. Use professional language but make it accessible."""
            },
            {
                "role": "user",
                "content": f"""
                Analyze the following company: {ticker}
                
                Financial Metrics:
                - Market Cap: ${company_data['Market Cap']:.2f}B
                - P/E Ratio: {company_data['P/E']:.2f}
                - ROE: {company_data['ROE']:.2%}
                - Debt/Equity: {company_data['Debt/Equity']:.2f}
                - Current Ratio: {company_data['Current Ratio']:.2f}
                - Operating Margin: {company_data['Operating Margin']:.2%}
                - Net Profit Margin: {company_data['Net Profit Margin']:.2%}
                - Beta: {company_data['Beta']:.2f}
                - Revenue Growth: {company_data['Revenue Growth']:.2%}
                - Z-Score: {company_data['Z_Score']:.2f}
                - Risk Level: {company_data['Risk_Level']}
                - Sector: {company_data['Sector']}
                - Growth/Value: {company_data['Growth_Value']}
                
                Please provide:
                1. Overall financial health assessment
                2. Key strengths and weaknesses
                3. Risk analysis
                4. Investment recommendation
                5. Sector comparison insights
                """
            }
        ]
        
        return self.llm_client.sync_generate_response(messages)
    
    def generate_portfolio_insights(self, df: pd.DataFrame, selected_tickers: List[str]) -> str:
        """Generate portfolio-level insights using LLM"""
        portfolio_data = df[df['Ticker'].isin(selected_tickers)]
        
        portfolio_summary = f"""
        Portfolio Summary:
        - Companies: {len(portfolio_data)}
        - Sectors: {', '.join(portfolio_data['Sector'].unique())}
        - Risk distribution: {portfolio_data['Risk_Level'].value_counts().to_dict()}
        - Average P/E: {portfolio_data['P/E'].mean():.2f}
        - Average ROE: {portfolio_data['ROE'].mean():.2%}
        - Average Beta: {portfolio_data['Beta'].mean():.2f}
        - Geographic distribution: {portfolio_data['Country'].value_counts().to_dict()}
        """
        
        messages = [
            {
                "role": "system",
                "content": """You are a portfolio manager with deep expertise in asset allocation, 
                risk management, and diversification strategies. Provide comprehensive portfolio analysis 
                and optimization recommendations."""
            },
            {
                "role": "user",
                "content": f"""
                Analyze this portfolio composition:
                {portfolio_summary}
                
                Individual holdings:
                {portfolio_data[['Ticker', 'Sector', 'Risk_Level', 'P/E', 'ROE', 'Beta']].to_string()}
                
                Please provide:
                1. Portfolio diversification analysis
                2. Risk assessment and concentration risks
                3. Sector allocation recommendations
                4. Geographic diversification insights
                5. Suggested optimizations
                6. Expected portfolio characteristics (risk/return profile)
                """
            }
        ]
        
        return self.llm_client.sync_generate_response(messages)
    
    def generate_market_insights(self, df: pd.DataFrame) -> str:
        """Generate market-wide insights using LLM"""
        market_summary = f"""
        Market Analysis Data:
        - Total companies analyzed: {len(df)}
        - Sector distribution: {df['Sector'].value_counts().to_dict()}
        - Country distribution: {df['Country'].value_counts().to_dict()}
        - Risk levels: {df['Risk_Level'].value_counts().to_dict()}
        - Growth vs Value: {df['Growth_Value'].value_counts().to_dict()}
        - Average metrics across all companies:
          * P/E: {df['P/E'].mean():.2f}
          * ROE: {df['ROE'].mean():.2%}
          * Debt/Equity: {df['Debt/Equity'].mean():.2f}
          * Revenue Growth: {df['Revenue Growth'].mean():.2%}
        """
        
        messages = [
            {
                "role": "system",
                "content": """You are a chief market strategist with expertise in macroeconomic analysis, 
                sector trends, and market cycles. Provide strategic market insights and investment themes."""
            },
            {
                "role": "user",
                "content": f"""
                Based on this cross-sectional analysis of companies:
                {market_summary}
                
                Please provide:
                1. Current market themes and trends
                2. Sector-specific opportunities and risks
                3. Geographic market insights
                4. Valuation analysis across sectors
                5. Investment strategy recommendations
                6. Risk factors to monitor
                """
            }
        ]
        
        return self.llm_client.sync_generate_response(messages)
    
    def explain_ml_results(self, df: pd.DataFrame, model_type: str) -> str:
        """Generate explanations for ML model results"""
        if model_type == "risk_classification":
            risk_dist = df['Risk_Level'].value_counts().to_dict()
            context = f"Risk classification results: {risk_dist}"
        elif model_type == "bankruptcy_prediction":
            z_score_stats = df['Z_Score'].describe().to_dict()
            context = f"Z-Score distribution: {z_score_stats}"
        elif model_type == "growth_value":
            gv_dist = df['Growth_Value'].value_counts().to_dict()
            context = f"Growth/Value classification: {gv_dist}"
        else:
            context = "General ML results"
        
        messages = [
            {
                "role": "system",
                "content": """You are a machine learning expert specializing in financial applications. 
                Explain ML results in business terms that investment professionals can understand and act upon."""
            },
            {
                "role": "user",
                "content": f"""
                Explain the following ML model results for {model_type}:
                {context}
                
                Dataset context:
                {self.create_financial_context(df)}
                
                Please provide:
                1. What the model results mean in business terms
                2. How to interpret the classifications/predictions
                3. Practical applications for investment decisions
                4. Limitations and considerations
                5. Next steps for implementation
                """
            }
        ]
        
        return self.llm_client.sync_generate_response(messages)

# Enhanced Gradio App with LLM Integration
def create_sample_data():
    """Create sample financial data (same as before)"""
    np.random.seed(42)
    
    tickers = [
        'AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA', 'TSLA', 'JPM', 'BAC', 'WFC',
        'JNJ', 'PFE', 'XOM', 'CVX', 'WMT', 'HD', 'HSBA.L', 'BP.L', 'SAP.DE',
        'VOW3.DE', '7203.T', 'BABA', 'INFY.NS', 'TCS.NS', 'RELIANCE.NS'
    ]
    
    data = []
    for ticker in tickers:
        market_cap = np.random.lognormal(15, 2)
        pe_ratio = np.random.gamma(2, 8)
        roe = np.random.normal(0.15, 0.1)
        debt_equity = np.random.gamma(1, 0.5)
        current_ratio = np.random.normal(1.5, 0.5)
        operating_margin = np.random.normal(0.2, 0.15)
        net_profit_margin = np.random.normal(0.1, 0.08)
        beta = np.random.normal(1.0, 0.3)
        revenue_growth = np.random.normal(0.08, 0.15)
        
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
    """Calculate Altman Z-Score"""
    z_scores = []
    for _, row in df.iterrows():
        working_capital_ratio = row['Current Ratio'] - 1
        retained_earnings_ratio = row['ROE'] * 0.1
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
    """Classify risk levels"""
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
    """Classify growth vs value"""
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

def main_analysis():
    """Main analysis function"""
    df = create_sample_data()
    df = calculate_altman_z_score(df)
    df = classify_risk_levels(df)
    df = classify_growth_value(df)
    return df

# LLM Integration Examples and Use Cases
def create_llm_integration_guide():
    """Create comprehensive LLM integration guide"""
    guide = """
    # 🤖 LLM Integration with LM Studio - Complete Guide
    
    ## 🚀 Setup Instructions
    
    ### 1. Install LM Studio
    ```bash
    # Download from: https://lmstudio.ai/
    # Install and launch LM Studio
    # Download a financial/business model (e.g., Mistral-7B, Llama-2-13B)
    ```
    
    ### 2. Start Local Server
    ```bash
    # In LM Studio:
    # 1. Go to "Local Server" tab
    # 2. Select your model
    # 3. Start server (default: localhost:1234)
    ```
    
    ### 3. Install Additional Dependencies
    ```bash
    pip install aiohttp asyncio
    ```
    
    ## 💡 Advanced Use Cases
    
    ### 1. 📊 Intelligent Financial Analysis
    - **Natural Language Queries**: "Which tech stocks have the best ROE?"
    - **Automated Report Generation**: Generate investment research reports
    - **Risk Assessment**: Explain risk factors in plain English
    - **Sector Comparison**: Compare companies across sectors with context
    
    ### 2. 🎯 Portfolio Optimization
    - **Asset Allocation**: Get AI-powered allocation recommendations
    - **Risk Management**: Identify portfolio concentration risks
    - **Diversification**: Suggest optimal diversification strategies
    - **Performance Attribution**: Explain portfolio performance drivers
    
    ### 3. 📈 Market Intelligence
    - **Trend Analysis**: Identify market trends from financial data
    - **Thematic Investing**: Discover investment themes and opportunities
    - **Scenario Analysis**: Model different market scenarios
    - **Comparative Analysis**: Compare companies, sectors, or markets
    
    ### 4. 🔍 ML Model Interpretation
    - **Feature Importance**: Explain which factors drive predictions
    - **Model Validation**: Assess model reliability and limitations
    - **Business Translation**: Convert technical results to business insights
    - **Actionable Recommendations**: Generate investment actions from ML results
    
    ## 🛠️ Technical Implementation
    
    ### A. Real-time Financial Chat
    ```python
    def financial_chatbot(query, df):
        context = create_financial_context(df)
        messages = [
            {"role": "system", "content": f"Financial data context: {context}"},
            {"role": "user", "content": query}
        ]
        return llm_client.sync_generate_response(messages)
    ```
    
    ### B. Automated Report Generation
    ```python
    def generate_investment_report(tickers, df):
        companies = df[df['Ticker'].isin(tickers)]
        return analyzer.generate_portfolio_insights(df, tickers)
    ```
    
    ### C. Risk Alert System
    ```python
    def check_risk_alerts(df):
        risky_companies = df[df['Risk_Level'] == 'Risky']
        return analyzer.explain_ml_results(risky_companies, 'risk_classification')
    ```
    
    ## 📋 Practical Applications
    
    ### 1. Investment Research Automation
    - Generate detailed company analysis reports
    - Compare investment opportunities
    - Identify undervalued stocks
    - Assess sector rotation opportunities
    
    ### 2. Risk Management
    - Early warning system for portfolio risks
    - Stress testing scenarios
    - Regulatory compliance reporting
    - ESG risk assessment
    
    ### 3. Client Communication
    - Generate client-friendly investment summaries
    - Explain complex financial concepts
    - Create personalized investment recommendations
    - Automate periodic portfolio reviews
    
    ### 4. Market Research
    - Trend identification and analysis
    - Competitive intelligence
    - Sector deep dives
    - Macroeconomic impact assessment
    
    ## 🔧 Integration Patterns
    
    ### Pattern 1: Query-Response System
    ```python
    # Natural language financial queries
    user_query = "What are the best performing tech stocks?"
    response = financial_chatbot(user_query, df)
    ```
    
    ### Pattern 2: Automated Analysis Pipeline
    ```python
    # Batch processing for multiple companies
    for ticker in portfolio_tickers:
        analysis = analyzer.analyze_company_with_llm(ticker, df)
        save_analysis_report(ticker, analysis)
    ```
    
    ### Pattern 3: Real-time Monitoring
    ```python
    # Continuous monitoring with LLM insights
    while True:
        new_data = fetch_latest_data()
        alerts = check_risk_alerts(new_data)
        if alerts:
            send_notification(alerts)
        time.sleep(3600)  # Check hourly
    ```
    
    ## 🎨 Advanced Features
    
    ### 1. Multi-modal Analysis
    - Combine financial data with news sentiment
    - Integrate technical chart analysis
    - Cross-reference with economic indicators
    
    ### 2. Personalized Insights
    - Adapt to user's investment style
    - Consider risk tolerance and preferences
    - Generate custom investment strategies
    
    ### 3. Real-time Market Commentary
    - Generate live market updates
    - Explain market movements
    - Provide trading insights
    
    ## 🚨 Best Practices
    
    ### Security & Privacy
    - Keep sensitive financial data local
    - Use secure API connections
    - Implement proper authentication
    - Regular security audits
    
    ### Performance Optimization
    - Cache frequent queries
    - Use async processing for batch operations
    - Implement request rate limiting
    - Monitor API response times
    
    ### Quality Control
    - Validate LLM responses
    - Cross-check with traditional analysis
    - Implement confidence scoring
    - Regular model performance review
    """
    
    return guide

def create_enhanced_gradio_app():
    """Create enhanced Gradio app with LLM integration"""
    df = main_analysis()
    
    # Initialize LLM client
    llm_client = LMStudioClient()
    analyzer = LLMFinancialAnalyzer(llm_client)
    
    with gr.Blocks(title="LLM-Enhanced Financial Analysis", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🤖 LLM-Enhanced Financial Analysis Dashboard")
        gr.Markdown("*Powered by LM Studio integration for intelligent financial insights*")
        
        with gr.Tab("🤖 AI Financial Assistant"):
            gr.Markdown("## Ask questions about your financial data")
            
            with gr.Row():
                with gr.Column():
                    query_input = gr.Textbox(
                        label="Ask a financial question",
                        placeholder="e.g., 'Which companies have the highest ROE?' or 'Analyze the risk in my portfolio'"
                    )
                    submit_btn = gr.Button("Get AI Analysis", variant="primary")
                
                with gr.Column():
                    ai_response = gr.Textbox(
                        label="AI Analysis",
                        lines=10,
                        placeholder="AI response will appear here..."
                    )
            
            def handle_query(query):
                if not query.strip():
                    return "Please enter a question."
                
                # Create context from the dataframe
                context = analyzer.create_financial_context(df)
                
                messages = [
                    {
                        "role": "system",
                        "content": f"""You are a financial analyst AI assistant. Use this data context: {context}
                        
                        Available companies: {', '.join(df['Ticker'].tolist())}
                        
                        Provide specific, actionable financial insights based on the data."""
                    },
                    {
                        "role": "user",
                        "content": query
                    }
                ]
                
                return llm_client.sync_generate_response(messages)
            
            submit_btn.click(handle_query, inputs=[query_input], outputs=[ai_response])
        
        with gr.Tab("🏢 AI Company Analysis"):
            with gr.Row():
                with gr.Column():
                    company_selector = gr.Dropdown(
                        choices=df['Ticker'].tolist(),
                        label="Select Company for AI Analysis",
                        value=df['Ticker'].iloc[0]
                    )
                    analyze_btn = gr.Button("Generate AI Analysis", variant="primary")
                
                with gr.Column():
                    company_analysis = gr.Textbox(
                        label="AI Company Analysis",
                        lines=15,
                        placeholder="Detailed AI analysis will appear here..."
                    )
            
            def analyze_company(ticker):
                if not ticker:
                    return "Please select a company."
                
                company_data = df[df['Ticker'] == ticker].iloc[0]
                return analyzer.analyze_company_with_llm(ticker, company_data)
            
            analyze_btn.click(analyze_company, inputs=[company_selector], outputs=[company_analysis])
        
        with gr.Tab("📊 Portfolio AI Insights"):
            with gr.Row():
                with gr.Column():
                    portfolio_selector = gr.CheckboxGroup(
                        choices=df['Ticker'].tolist(),
                        label="Select Portfolio Companies",
                        value=df['Ticker'].tolist()[:5]
                    )
                    portfolio_btn = gr.Button("Generate Portfolio Analysis", variant="primary")
                
                with gr.Column():
                    portfolio_analysis = gr.Textbox(
                        label="AI Portfolio Analysis",
                        lines=15,
                        placeholder="Portfolio insights will appear here..."
                    )
            
            def analyze_portfolio(selected_tickers):
                if not selected_tickers:
                    return "Please select at least one company."
                
                return analyzer.generate_portfolio_insights(df, selected_tickers)
            
            portfolio_btn.click(analyze_portfolio, inputs=[portfolio_selector], outputs=[portfolio_analysis])
        
        with gr.Tab("🎯 ML Model Explanation"):
            with gr.Row():
                with gr.Column():
                    model_selector = gr.Dropdown(
                        choices=["risk_classification", "bankruptcy_prediction", "growth_value"],
                        label="Select ML Model to Explain",
                        value="risk_classification"
                    )
                    explain_btn = gr.Button("Get AI Explanation", variant="primary")
                
                with gr.Column():
                    model_explanation = gr.Textbox(
                        label="AI Model Explanation",
                        lines=12,
                        placeholder="ML model explanation will appear here..."
                    )
            
            def explain_model(model_type):
                return analyzer.explain_ml_results(df, model_type)
            
            explain_btn.click(explain_model, inputs=[model_selector], outputs=[model_explanation])
        
        with gr.Tab("🌍 Market Intelligence"):
            with gr.Row():
                with gr.Column():
                    market_btn = gr.Button("Generate Market Insights", variant="primary")
                
                with gr.Column():
                    market_insights = gr.Textbox(
                        label="AI Market Analysis",
                        lines=15,
                        placeholder="Market insights will appear here..."
                    )
            
            def generate_market_analysis():
                return analyzer.generate_market_insights(df)
            
            market_btn.click(generate_market_analysis, outputs=[market_insights])
        
        with gr.Tab("📚 LLM Integration Guide"):
            guide_content = create_llm_integration_guide()
            gr.Markdown(guide_content)
        
        with gr.Tab("⚙️ LM Studio Setup"):
            gr.Markdown("""
            # 🛠️ LM Studio Configuration
            
            ## Current Connection Status
            - **Server URL**: http://localhost:1234
            - **API Endpoint**: /v1/chat/completions
            
            ## Setup Steps:
            1. **Download LM Studio**: Visit https://lmstudio.ai/
            2. **Install Model**: Choose a model (recommended: Mistral-7B or Llama-2-13B)
            3. **Start Server**: Go to "Local Server" tab and start server
            4. **Test Connection**: Use the test button below
            
            ## Recommended Models for Financial Analysis:
            - **Mistral-7B-Instruct**: Great for general financial analysis
            - **Llama-2-13B-Chat**: Better for complex reasoning
            - **CodeLlama-13B**: Good for technical analysis
            - **Zephyr-7B**: Fast and efficient for real-time queries
            """)
            
            test_btn = gr.Button("Test LM Studio Connection", variant="secondary")
            connection_status = gr.Textbox(label="Connection Status", lines=3)
            
            def test_connection():
                test_messages = [
                    {"role": "user", "content": "Hello, are you working correctly?"}
                ]
                response = llm_client.sync_generate_response(test_messages)
                
                if "Error" in response or "Connection error" in response:
                    return f"❌ Connection failed: {response}"
                else:
                    return f"✅ Connection successful! Response: {response}"
            
            test_btn.click(test_connection, outputs=[connection_status])
    
    return demo

# Launch the enhanced app
if __name__ == "__main__":
    demo = create_enhanced_gradio_app()
    demo.launch(share=True)