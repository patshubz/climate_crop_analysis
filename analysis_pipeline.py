import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.ensemble import RandomForestRegressor
import ruptures as rpt
from statsmodels.tsa.arima.model import ARIMA
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def load_and_preprocess_data():
    climate_df = pd.read_csv('climate_data.csv', parse_dates=['date'])
    crop_df = pd.read_csv('crop_data.csv', parse_dates=['date', 'planting_date', 'harvest_date'])
    region_df = pd.read_csv('region_data.csv')
    merged_df = pd.merge(crop_df, climate_df, on=['date', 'region_id'])
    merged_df = pd.merge(merged_df, region_df, on='region_id')
    merged_df = merged_df.ffill()
    numeric_cols = merged_df.select_dtypes(include=[np.number]).columns
    z_scores = np.abs(stats.zscore(merged_df[numeric_cols], nan_policy='omit'))
    merged_df = merged_df[(z_scores < 3).all(axis=1)]
    return merged_df

def statistical_analysis(df):
    numeric_df = df.select_dtypes(include=[np.number])
    correlation_matrix = numeric_df.corr()[['yield']].sort_values(by='yield', ascending=False)
    df['year'] = df['date'].dt.year
    temp_trend = df.groupby('year')['average_temperature'].mean()
    precip_trend = df.groupby('year')['precipitation'].mean()
    extreme_impact = df.groupby('extreme_weather_events')['yield'].mean()
    crop_impact = df.groupby('crop_type')['yield'].mean()
    return correlation_matrix, temp_trend, precip_trend, extreme_impact, crop_impact

def prepare_time_series_data(df, variable, region_id=None):
    if region_id is not None:
        df = df[df['region_id'] == region_id]
    else:
        df = df.groupby('date')[variable].mean().reset_index()

    if df.empty or df['date'].isnull().all():
        return pd.Series(dtype=float)

    start_date = df['date'].min()
    end_date = df['date'].max()
    if pd.isna(start_date) or pd.isna(end_date):
        return pd.Series(dtype=float)

    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    complete_df = pd.DataFrame(index=date_range)
    complete_df[variable] = df.set_index('date')[variable]
    complete_df[variable] = complete_df[variable].interpolate(method='time')
    return complete_df[variable]

def time_series_analysis(df):
    temp_series = prepare_time_series_data(df, 'average_temperature')
    yield_series = prepare_time_series_data(df, 'yield')
    
    def decompose_time_series(series):
        series = series.interpolate(method='time')
        # Check if we have enough data for seasonal decomposition
        if len(series) >= 730:  # 2 complete cycles
            decomposition = seasonal_decompose(series, model='additive', period=365)
        else:
            # Create a simple trend decomposition instead
            trend = series.rolling(window=3, min_periods=1).mean()
            seasonal = pd.Series(0, index=series.index)  # No seasonal component
            resid = series - trend
            # Create a dummy decomposition object with similar attributes
            decomposition = type('DummyDecomposition', (), {
                'observed': series,
                'trend': trend,
                'seasonal': seasonal,
                'resid': resid
            })
        return decomposition

    temp_decomp = decompose_time_series(temp_series)
    yield_decomp = decompose_time_series(yield_series)
    
    # Calculate lag correlations
    lag_correlations = {}
    for region_id in df['region_id'].unique():
        region_df = df[df['region_id'] == region_id]
        region_lags = {}
        for lag in range(1, min(31, len(region_df) - 1)):  # Ensure we don't exceed data length
            region_lags[lag] = region_df['yield'].corr(region_df['average_temperature'].shift(lag))
        lag_correlations[region_id] = region_lags
    
    return temp_decomp, yield_decomp, lag_correlations

def predictive_modeling(df):
    numeric_features = ['average_temperature', 'precipitation', 'humidity',
                        'wind_speed', 'solar_radiation', 'soil_moisture',
                        'fertilizer_usage', 'elevation']
    categorical_features = ['crop_type', 'soil_type', 'irrigation_system']
    df_encoded = pd.get_dummies(df, columns=categorical_features)
    all_features = numeric_features + [col for col in df_encoded.columns
                                       if any(col.startswith(cat) for cat in categorical_features)]
    X = df_encoded[all_features]
    y = df_encoded['yield']
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    tscv = TimeSeriesSplit(n_splits=5)
    scores = cross_val_score(model, X, y, cv=tscv, scoring='neg_mean_squared_error')
    print(f'Mean Squared Error: {-np.mean(scores):.2f}')
    model.fit(X, y)
    feature_importance = pd.DataFrame({
        'feature': all_features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    return model, feature_importance, all_features, numeric_features, categorical_features

def detect_regime_changes(series, pen=10):
    """
    Detect regime changes in time series data using PELT algorithm.
    Returns empty list if insufficient data.
    """
    # Check if we have enough data points (at least 5 for meaningful change detection)
    if len(series) < 5:
        print("Warning: Insufficient data for regime change detection")
        return []
    
    try:
        # Convert series to numpy array and handle NaN values
        signal = series.values.astype('float64')
        signal = np.nan_to_num(signal, nan=np.nanmean(signal))
        
        # Reshape signal to 2D array as required by ruptures
        signal = signal.reshape(-1, 1)
        
        # Use a more robust model ('l2' instead of 'rbf')
        algo = rpt.Pelt(model="l2").fit(signal)
        
        # Adjust penalty parameter based on data length
        adaptive_pen = pen * np.log(len(signal))
        result = algo.predict(pen=adaptive_pen)
        
        # Filter out spurious change points
        if len(result) > 1:  # Only keep points if we detected any changes
            return result
        return []
        
    except Exception as e:
        print(f"Warning: Error in regime change detection: {str(e)}")
        return []

def forecast_yield(yield_series, steps=30):
    model = ARIMA(yield_series, order=(1,1,1))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=steps)
    return forecast

def compare_model_performance_by_region(df, features, target):
    results = {}
    for region_id, group_df in df.groupby('region_id'):
        if len(group_df) < 30: continue
        X, y = group_df[features], group_df[target]
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        tscv = TimeSeriesSplit(n_splits=3)
        scores = cross_val_score(model, X, y, cv=tscv, scoring='neg_mean_squared_error')
        results[region_id] = np.mean(-scores)
    return results

def compare_model_performance_by_crop(df, numeric_features, categorical_features, target):
    results = {}
    for crop_type, group_df in df.groupby('crop_type'):
        if len(group_df) < 30: continue
        group_encoded = pd.get_dummies(group_df, columns=categorical_features)
        all_features = numeric_features + [col for col in group_encoded.columns
                                           if any(col.startswith(cat) for cat in categorical_features)]
        X, y = group_encoded[all_features], group_encoded[target]
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        tscv = TimeSeriesSplit(n_splits=3)
        scores = cross_val_score(model, X, y, cv=tscv, scoring='neg_mean_squared_error')
        results[crop_type] = np.mean(-scores)
    return results

def optimize_fertilizer(df, model, features, region_id, crop_type):
    base = df[(df['region_id'] == region_id) & (df['crop_type'] == crop_type)]
    if base.empty: return None
    median_vals = base.median(numeric_only=True)
    soil_type = base['soil_type'].mode().iloc[0] if 'soil_type' in base and not base['soil_type'].isnull().all() else 'loamy'
    irrigation_system = base['irrigation_system'].mode().iloc[0] if 'irrigation_system' in base and not base['irrigation_system'].isnull().all() else 'drip'
    test_rows = []
    fert_range = np.linspace(50, 200, 20)
    for fert in fert_range:
        row = median_vals.copy()
        row['fertilizer_usage'] = fert
        row['region_id'] = region_id
        row['crop_type'] = crop_type
        row['soil_type'] = soil_type
        row['irrigation_system'] = irrigation_system
        test_row_df = pd.DataFrame([row])
        test_row_encoded = pd.get_dummies(test_row_df, columns=['crop_type', 'soil_type', 'irrigation_system'])
        for col in features:
            if col not in test_row_encoded.columns:
                test_row_encoded[col] = 0
        test_row_encoded = test_row_encoded[features]
        test_rows.append(test_row_encoded.iloc[0])
    test_rows_df = pd.DataFrame(test_rows, columns=features)
    preds = model.predict(test_rows_df)
    best_idx = np.argmax(preds)
    return fert_range[best_idx], preds[best_idx]

def create_visualizations(df, correlation_matrix, temp_trend, precip_trend,
                          feature_importance, crop_impact, temp_decomp, lag_correlations):
    df['year'] = df['date'].dt.year
    # Interactive map with plotly express
    fig_map = px.scatter_geo(df, lat='latitude', lon='longitude',
                             color='yield', size='yield',
                             hover_name='region_id',
                             animation_frame='year',
                             projection="natural earth",
                             title='Crop Yields by Region')
    # Correlation heatmap
    fig_heatmap = px.imshow(correlation_matrix,
                            labels=dict(color="Correlation"),
                            title='Correlation with Crop Yield')
    # Time series trends
    fig_trends = make_subplots(rows=2, cols=1)
    fig_trends.add_trace(go.Scatter(x=temp_trend.index, y=temp_trend.values,name='Temperature Trend'), row=1,col=1)
    fig_trends.add_trace(go.Scatter(x=precip_trend.index, y=precip_trend.values,name='Precipitation Trend'), row=2,col=1)
    fig_trends.update_layout(title='Climate Trends Over Time')
    # Feature importance
    fig_features = px.bar(feature_importance, x='feature', y='importance', title='Feature Importance for Yield Prediction')
    # Crop impact
    fig_crop = px.bar(x=crop_impact.index, y=crop_impact.values, title='Average Yield by Crop Type')
    # Decomposition plots
    fig_decomp_temp = make_subplots(rows=4, cols=1)
    fig_decomp_temp.add_trace(go.Scatter(x=temp_decomp.observed.index, y=temp_decomp.observed, name='Observed'), row=1, col=1)
    fig_decomp_temp.add_trace(go.Scatter(x=temp_decomp.trend.index, y=temp_decomp.trend, name='Trend'), row=2, col=1)
    fig_decomp_temp.add_trace(go.Scatter(x=temp_decomp.seasonal.index, y=temp_decomp.seasonal, name='Seasonal'), row=3, col=1)
    fig_decomp_temp.add_trace(go.Scatter(x=temp_decomp.resid.index, y=temp_decomp.resid, name='Residual'), row=4, col=1)
    fig_decomp_temp.update_layout(title='Temperature Decomposition')
    # Lag correlation plot
    fig_lag = go.Figure()
    for region_id, lags in lag_correlations.items():
        fig_lag.add_trace(go.Scatter(x=list(lags.keys()), y=list(lags.values()), name=f'Region {region_id}'))
    fig_lag.update_layout(title='Lag Correlations by Region', xaxis_title='Lag (days)', yaxis_title='Correlation')
    return fig_map, fig_heatmap, fig_trends, fig_features, fig_crop, fig_decomp_temp, fig_lag

def main():
    df = load_and_preprocess_data()
    correlation_matrix, temp_trend, precip_trend, extreme_impact, crop_impact = statistical_analysis(df)
    temp_decomp, yield_decomp, lag_correlations = time_series_analysis(df)
    model, feature_importance, all_features, numeric_features, categorical_features = predictive_modeling(df)

    # Regime change detection
    temp_regimes = detect_regime_changes(temp_trend)
    if temp_regimes:
        change_years = [temp_trend.index[i-1] for i in temp_regimes if i > 0 and i < len(temp_trend)]
        print(f"\nTemperature regime change points (years): {change_years}")
    else:
        print("\nNo significant temperature regime changes detected")
    
    # Yield forecasting
    yield_series = prepare_time_series_data(df, 'yield')
    yield_forecast = forecast_yield(yield_series, steps=30)
    print(f"\nYield forecast for next 30 days:\n{yield_forecast}")
    # Model performance by region and crop
    region_perf = compare_model_performance_by_region(df, all_features, 'yield')
    crop_perf = compare_model_performance_by_crop(df, numeric_features, categorical_features, 'yield')
    print(f"\nModel RMSE by region: {region_perf}")
    print(f"Model RMSE by crop type: {crop_perf}")
    # Fertilizer optimization
    print("\nFertilizer optimization (best usage for max yield):")
    for region_id in df['region_id'].unique():
        for crop_type in df['crop_type'].unique():
            result = optimize_fertilizer(df, model, all_features, region_id, crop_type)
            if result:
                best_fert, best_yield = result
                print(f"Region {region_id}, Crop {crop_type}: Fertilizer {best_fert:.1f} -> Predicted Yield {best_yield:.2f}")
    # Visualizations
    fig_map, fig_heatmap, fig_trends, fig_features, fig_crop, fig_decomp_temp, fig_lag = create_visualizations(
        df, correlation_matrix, temp_trend, precip_trend,
        feature_importance, crop_impact, temp_decomp, lag_correlations
    )
    fig_map.write_html("map_visualization.html")
    fig_heatmap.write_html("correlation_heatmap.html")
    fig_trends.write_html("climate_trends.html")
    fig_features.write_html("feature_importance.html")
    fig_crop.write_html("crop_impact.html")
    fig_decomp_temp.write_html("temperature_decomposition.html")
    fig_lag.write_html("lag_correlations.html")
    # Print key findings
    print("\nKey Findings:")
    print(f"1. Most important features for yield prediction: {feature_importance['feature'].iloc[:3].tolist()}")
    print(f"2. Average temperature trend: {temp_trend.pct_change().mean():.2%} per year")
    print(f"3. Crop type impact on yield:\n{crop_impact}")
    print(f"4. Temperature seasonal pattern strength: {temp_decomp.seasonal.std():.2f}")
    print("\nLag Correlation Findings:")
    for region_id, lags in lag_correlations.items():
        max_lag = max(lags.items(), key=lambda x: abs(x[1]))
        print(f"Region {region_id}: Maximum correlation at {max_lag[0]} days lag (correlation: {max_lag[1]:.2f})")

if __name__ == "__main__":
    main()
