# scripts/scrape_data.py
import yfinance as yf
import pandas as pd
import time

# International + US tickers
tickers = [
    # US (Actively Traded)
    # Tech
    'AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA', 'ADBE', 'ORCL', 'INTC', 'CSCO', 'IBM',
    # Financials
    'JPM', 'BAC', 'WFC', 'GS', 'C', 'MS', 'AXP', 'V', 'MA',
    # Consumer Discretionary
    'TSLA', 'GM', 'F', 'HD', 'WMT', 'TGT', 'MCD', 'SBUX', 'NKE', 'KO',
    # Healthcare
    'JNJ', 'PFE', 'MRK', 'UNH', 'ABBV', 'AMGN', 'LLY', 'GILD',
    # Energy
    'XOM', 'CVX', 'BP', 'TOT', 'COP', 'SLB', # SLB is Schlumberger (Oilfield Services)
    # Industrials
    'GE', 'CAT', 'UPS', 'FDX', 'BA', 'HON', # Honeywell (Diversified Industrial)
    # Utilities
    'NEE', 'DUK', 'SO', # NextEra Energy, Duke Energy, Southern Company
    # Real Estate
    'PLD', 'SPG', # Prologis (REIT), Simon Property Group (Retail REIT)
    # Materials
    'DD', 'LIN', # DuPont (Chemicals), Linde (Industrial Gases)
    # Communications Services
    'VZ', 'T', 'CMCSA', # Verizon, AT&T, Comcast
    # Consumer Staples
    'PG', 'KO', 'PEP', 'KHC', # Procter & Gamble, Coca-Cola, PepsiCo, Kraft Heinz
    # Misc/Small Cap / Recent IPOs (last 12-18 months for illustrative purposes)
    'AMC', 'GME', 'PLTR', 'SNAP', 'COIN',
    'ARM', # Arm Holdings plc (Semiconductor IP - recent major IPO, also UK-based)
    'GEHC', # GE HealthCare Technologies Inc. (Spinoff from GE)
    'KRTX', # Karat Packaging Inc. (Manufacturing/Packaging - smaller cap, example)
    'ABNB', # Airbnb (Travel/Tech - 2020 IPO, but widely known)
    'RIVN', # Rivian Automotive (EVs - 2021 IPO, significant for auto sector)

    # US (Bankruptcies/Delistings - Tickers might be OTC or inactive)
    'PRTYQ', # Party City Holdco Inc.
    'SAVE',  # Spirit Airlines (still trading but distressed; use SAVEQ if delisted for bankruptcy)
    'TUPBQ', # Tupperware Brands Corporation
    'BIG',   # Big Lots (still trading but distressed; use BIGQ if delisted for bankruptcy)
    'RUE',   # rue21 Inc. (likely very inactive or not found)
    'JOANQ', # Joann Inc. (filed Chapter 11 Jan 2025)
    'RTHTQ', # Rite Aid Corp. (filed Chapter 11 Oct 2023; often RADCQ or RTH)

    # UK (Actively Traded)
    'HSBA.L', 'BP.L', 'AZN.L', 'GSK.L', # Financials, Energy, Pharma
    'ULVR.L', # Unilever (Consumer Staples)
    'SHEL.L', # Shell plc (Energy - previously RDSB.L)
    'VOD.L', # Vodafone Group plc (Telecommunications)
    'LLOY.L', # Lloyds Banking Group plc (Financials)
    'RIO.L', # Rio Tinto plc (Mining/Materials)
    'NG.L', # National Grid plc (Utilities)
    'LAND.L', # Land Securities Group plc (Real Estate)
    'SGRO.L', # Segro plc (Real Estate)

    # Germany (Actively Traded)
    'VOW3.DE', 'BMW.DE', 'SAP.DE', 'BAS.DE', # Auto, Software, Chemicals
    'ALV.DE', # Allianz SE (Insurance/Financials)
    'SIE.DE', # Siemens AG (Industrials/Technology)
    'DTE.DE', # Deutsche Telekom AG (Telecommunications)
    'DBK.DE', # Deutsche Bank AG (Financials)
    'ADS.DE', # Adidas AG (Consumer Discretionary)
    'BAYN.DE', # Bayer AG (Healthcare/Chemicals)
    'RWE.DE', # RWE AG (Utilities/Energy)
    'HEI.DE', # Heidelberg Materials AG (Construction Materials)
    'FRE.DE', # Fresenius SE & Co. KGaA (Healthcare)

    # Germany (Bankruptcies/Delistings - Tickers might be inactive)
    'VAR1.DE', # Varta AG (delisted from Frankfurt March 2025)
    # Galeria Karstadt Kaufhof (often part of a larger group or private entity, complex ticker)

    # Japan (Actively Traded)
    '7203.T', # Toyota Motor Corp (Auto)
    '6758.T', # Sony Group Corp (Electronics/Tech)
    '9984.T', # SoftBank Group Corp (Tech Investment)
    '9432.T', # Nippon Telegraph and Telephone Corp (Telecoms)
    '8058.T', # Mitsubishi Corp (Trading/Diversified)
    '8306.T', # Mitsubishi UFJ Financial Group, Inc. (Financials)
    '4519.T', # Chugai Pharmaceutical Co., Ltd. (Healthcare)
    '6098.T', # Recruit Holdings Co., Ltd. (HR/Tech)
    '6954.T', # Fanuc Corp (Robotics/Industrials)
    '9020.T', # East Japan Railway Co. (Rail Transport/Industrials)
    '8802.T', # Mitsubishi Estate Co., Ltd. (Real Estate)
    '8604.T', # Nomura Holdings, Inc. (Financials)

    # China (US-listed, Actively Traded - often ADRs)
    'BABA', 'JD', 'PDD', 'NIO', 'LI', # E-commerce, EVs
    'TCEHY', # Tencent Holdings Ltd. (Tech/Entertainment - ADR)
    'BIDU', # Baidu Inc. (Tech/AI)
    'KWEB', # KraneShares CSI China Internet ETF (to cover multiple internet companies)
    'BILI', # Bilibili Inc. (Online Entertainment)
    'XPEV', # XPeng Inc. (EVs)
    'ZIM', # Zim Integrated Shipping Services Ltd. (Israel-based, but active in China/Asia shipping)

    # India (Actively Traded - use .NS for NSE or .BO for BSE)
    'INFY.NS', 'TCS.NS', 'RELIANCE.NS', # IT Services, Diversified Conglomerate
    'HDFCBANK.NS', # HDFC Bank Ltd. (Financials)
    'ICICIBANK.NS', # ICICI Bank Ltd. (Financials)
    'BHARTIARTL.NS', # Bharti Airtel Ltd. (Telecommunications)
    'LT.NS', # Larsen & Toubro Ltd. (Engineering & Construction/Industrials)
    'TATASTEEL.NS', # Tata Steel Ltd. (Materials/Metals)
    'MARUTI.NS', # Maruti Suzuki India Ltd. (Automobiles)
    'ASIANPAINT.NS', # Asian Paints Ltd. (Chemicals/Consumer Goods)
    'HINDUNILVR.NS', # Hindustan Unilever Ltd. (Consumer Staples)
    'DRREDDY.NS', # Dr. Reddy's Laboratories Ltd. (Healthcare/Pharma)
    'ADANIPORTS.NS', # Adani Ports and Special Economic Zone Ltd. (Infrastructure)
    'POWERGRID.NS', # Power Grid Corporation of India Ltd. (Utilities)

    # India (Bankruptcies/Delistings - Tickers might be inactive or delisted)
    'FRTL.NS',    # Future Retail Ltd.
    'RELCAPITAL.NS', # Reliance Capital
    'SINTX.NS',   # Sintex Plastics Technology
    'AGCM.BO',    # Agrimony Commodities
    'JINDCOT.NS', # Jindal Cotex Ltd.
    'SKUMARSO.NS', # S Kumars Online Ltd.
    # Go First Airlines (ticker was for parent, no direct trading post-grounding)
]

data = []
for ticker in tickers:
    print(f"Fetching {ticker}")
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        data.append({
            'Ticker': ticker,
            'Market Cap': info.get('marketCap'),
            'P/E': info.get('trailingPE'),
            'ROE': info.get('returnOnEquity'),
            'Debt/Equity': info.get('debtToEquity'),
            'Current Ratio': info.get('currentRatio'),
            'Operating Margin': info.get('operatingMargins'),
            'Net Profit Margin': info.get('netMargins'),
            'Free Cash Flow': info.get('freeCashflow'),
            'Beta': info.get('beta'),
            'Revenue Growth': info.get('revenueGrowth'),
            'Gross Profits': info.get('grossProfits'),
            'Total Revenue': info.get('totalRevenue'),
            'Total Assets': info.get('totalAssets'),
            'Total Liabilities': info.get('totalLiab'),
            'EBIT': info.get('ebit'),
            'Working Capital': info.get('totalCurrentAssets', 0) - info.get('totalCurrentLiabilities', 0),
            'Retained Earnings': info.get('retainedEarnings'),
            'Total Equity': info.get('totalStockholderEquity')
        })
    except Exception as e:
        print(f"[!] Error fetching {ticker}: {e}")
    time.sleep(1)

df = pd.DataFrame(data)
df.to_csv("data/raw/financial_statements.csv", index=False)
print("✅ Saved to data/raw/financial_statements.csv")
