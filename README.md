# Energy-Price-Forecasting
Energy commodity forecasting pipeline integrating ARIMAX-GARCH, VECM-GARCH, and hybrid volatility-transmission models.
The pipeline automates data ingestion from public sources (NOAA, EIA, SEC, DOE, WH, etc.) and integrates into model training, rolling/expanding backtests, and visualizations.

# Core Featuresa

  - ARIMX-GARCH (1,1) for Natural Gas
  - ARIMAX(0,0,4) for Crude Oil
  - VECM-GARCH Hybrid for capturing NG-OL cointegration and volatility spillovers
  - Exogenous Integration via HDD/CDD (historical + forecasts) and FinBERT-based sentiment analysis
  -  
