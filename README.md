# Project Title: Climate Impact and Crop Yield Analysis

## Overview
This project provides a comprehensive Python data analysis pipeline to study the effects of climate change on crop yields across multiple regions over time. It integrates data cleaning, statistical analysis, time series modeling, machine learning predictions, regime change detection, and interactive visualizations to deliver actionable insights for agricultural planning and research.

## Features
- Loads and preprocesses multi-source CSV datasets (climate, crop, region)
- Calculates correlations and trends between climate variables and crop yield
- Decomposes time series data to extract seasonality and trends
- Detects climate regime shifts using robust change point algorithms
- Builds machine learning models to forecast crop yields, assesses feature importance
- Suggests optimal fertilizer usage to maximize yield based on model predictions
- Generates interactive visualizations: maps, heatmaps, trend plots, and dashboards
- Structured for modularity, scalability, and ease of adaptation to other datasets

## Requirements
- Python 3.8+
- Libraries:
  - pandas (≥1.3)
  - numpy
  - scipy
  - statsmodels
  - scikit-learn
  - ruptures
  - plotly

Install all dependencies via pip:
```bash
pip install pandas numpy scipy statsmodels scikit-learn ruptures plotly
```

## Usage
1. Place your CSV data files in the project directory:
   - `climate_data.csv`
   - `crop_data.csv`
   - `region_data.csv`

2. Run the main analysis script:
   ```bash
   python analysis_pipeline.py
   ```

3. Results and findings will be printed to the console.

4. Interactive visualizations are saved as HTML files:
   - `map_visualization.html`
   - `correlation_heatmap.html`
   - `climate_trends.html`
   - `feature_importance.html`
   - `crop_impact.html`
   - `temperature_decomposition.html`
   - `lag_correlations.html`

   Open these files in any modern web browser to explore the insights.

## Project Structure
```
climate_crop_analysis/
│
├── analysis_pipeline.py        # Main analysis and modeling code
├── test_analysis.py            # Test suite using pytest
├── climate_data.csv            # Climate variables time series data
├── crop_data.csv               # Crop yields and related measurements
├── region_data.csv             # Information about different regions
├── requirements.txt            # Python dependencies
└── README.md                   # Project overview and instructions
```

## Testing
Run tests with coverage reporting:
```bash
pytest --cov=analysis_pipeline --cov-report=term --cov-report=html
```
- Test reports show code coverage and help ensure correctness
- Tests cover data loading, statistical functions, modeling, and visualization generation

