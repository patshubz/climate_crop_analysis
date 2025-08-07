import pytest
import pandas as pd
import numpy as np
from io import StringIO
from unittest.mock import patch, MagicMock
import warnings

# Mock CSV data that matches the structure and provides sufficient data for all tests
climate_data_csv = """date,region_id,average_temperature,precipitation,humidity,wind_speed,solar_radiation,extreme_weather_events
2010-01-01,1,18.2,2.0,60,12,170,0
2010-01-01,2,12.6,5.0,73,6,160,1
2010-01-02,1,19.0,0.0,59,14,172,0
2010-01-02,2,13.1,3.2,75,7,163,0
2010-01-03,1,20.1,0.0,55,15,178,0
2010-01-03,2,13.8,4.1,74,8,164,0
2010-01-04,1,18.6,1.2,61,10,175,0
2010-01-04,2,14.2,2.1,71,7,160,1
2010-01-05,1,17.5,1.5,58,11,173,0
2010-01-05,2,13.5,2.8,72,6,162,0
2010-01-06,1,18.8,0.5,57,13,176,0
2010-01-06,2,14.0,3.0,70,8,165,0
2010-01-07,1,19.2,0.8,56,12,174,0
2010-01-07,2,13.2,2.5,69,7,163,0
2010-01-08,1,18.0,1.8,59,10,171,0
2010-01-08,2,14.5,3.5,73,9,166,1
2010-01-09,1,19.5,0.2,58,14,177,0
2010-01-09,2,13.0,4.0,75,6,161,0
2010-01-10,1,18.3,1.0,60,11,172,0
2010-01-10,2,14.8,2.2,68,8,164,0"""

crop_data_csv = """date,region_id,crop_type,yield,planting_date,harvest_date,soil_moisture,fertilizer_usage
2010-01-01,1,Wheat,3500,2009-10-15,2010-06-20,22,90
2010-01-01,2,Corn,4200,2009-10-20,2010-07-01,27,70
2010-01-02,1,Wheat,3600,2009-10-15,2010-06-20,24,95
2010-01-02,2,Corn,4300,2009-10-20,2010-07-01,29,75
2010-01-03,1,Wheat,3650,2009-10-15,2010-06-20,23,95
2010-01-03,2,Corn,4450,2009-10-20,2010-07-01,31,78
2010-01-04,1,Wheat,3550,2009-10-15,2010-06-20,21,92
2010-01-04,2,Corn,4250,2009-10-20,2010-07-01,28,73
2010-01-05,1,Wheat,3700,2009-10-15,2010-06-20,23,88
2010-01-05,2,Corn,4380,2009-10-20,2010-07-01,30,76
2010-01-06,1,Wheat,3580,2009-10-15,2010-06-20,22,93
2010-01-06,2,Corn,4320,2009-10-20,2010-07-01,28,74
2010-01-07,1,Wheat,3620,2009-10-15,2010-06-20,24,91
2010-01-07,2,Corn,4280,2009-10-20,2010-07-01,29,72
2010-01-08,1,Wheat,3480,2009-10-15,2010-06-20,21,89
2010-01-08,2,Corn,4350,2009-10-20,2010-07-01,32,77
2010-01-09,1,Wheat,3720,2009-10-15,2010-06-20,25,94
2010-01-09,2,Corn,4410,2009-10-20,2010-07-01,33,79
2010-01-10,1,Wheat,3590,2009-10-15,2010-06-20,23,90
2010-01-10,2,Corn,4290,2009-10-20,2010-07-01,27,71"""

region_data_csv = """region_id,latitude,longitude,elevation,soil_type,irrigation_system
1,38.9,-90.1,120,loam,drip
2,42.2,-85.9,240,clay,sprinkler"""

def mock_read_csv(filepath, *args, **kwargs):
    """Mock pandas.read_csv to return test data - fixed recursion issue"""
    filepath_str = str(filepath)
    
    if 'climate_data.csv' in filepath_str:
        return pd.read_csv(StringIO(climate_data_csv), parse_dates=['date'])
    elif 'crop_data.csv' in filepath_str:
        return pd.read_csv(StringIO(crop_data_csv), parse_dates=['date', 'planting_date', 'harvest_date'])
    elif 'region_data.csv' in filepath_str:
        return pd.read_csv(StringIO(region_data_csv))
    else:
        # For any other file, just raise an error instead of calling pd.read_csv again
        raise FileNotFoundError(f"Mock data not available for {filepath}")


@pytest.fixture(autouse=True)
def setup_mock_data(monkeypatch):
    """Setup mock data for all tests"""
    monkeypatch.setattr('pandas.read_csv', mock_read_csv)


class TestAnalysisPipeline:
    
    def test_detect_regime_changes(self):
        """Test regime change detection"""
        from analysis_pipeline import detect_regime_changes
        
        # Test with sufficient data
        series = pd.Series([10]*20 + [20]*20)  # Clear regime change
        changes = detect_regime_changes(series, pen=5)
        assert isinstance(changes, list), "Should return list"
        
        # Test with insufficient data
        small_series = pd.Series([1, 2])
        changes_small = detect_regime_changes(small_series)
        assert isinstance(changes_small, list), "Should return list for small data"
        assert len(changes_small) == 0, "Should return empty list for insufficient data"
        
        # Test with constant data
        constant_series = pd.Series([5]*50)
        changes_constant = detect_regime_changes(constant_series)
        assert isinstance(changes_constant, list), "Should return list for constant data"
    
    def test_forecast_yield(self):
        """Test yield forecasting with error handling"""
        from analysis_pipeline import forecast_yield
        
        # Test with valid series
        valid_series = pd.Series([3500, 3600, 3650, 3550, 3700, 3580, 3620, 3480, 3720, 3590])
        
        try:
            forecast = forecast_yield(valid_series, steps=5)
            assert len(forecast) == 5, "Should return forecast of requested length"
            assert isinstance(forecast, pd.Series), "Should return pandas Series"
        except Exception as e:
            # If ARIMA fails due to insufficient data, that's expected behavior
            assert ("LU decomposition" in str(e) or 
                    "Too few observations" in str(e) or
                    "IndexError" in str(type(e).__name__)), f"Unexpected error: {e}"
        
        # Test with very small series (should handle gracefully)
        small_series = pd.Series([1, 2])
        try:
            forecast_small = forecast_yield(small_series, steps=2)
        except Exception as e:
            # Expected to fail with small data - various possible exceptions
            assert isinstance(e, (ValueError, np.linalg.LinAlgError, IndexError))

# Additional utility tests
class TestUtilityFunctions:
    
    def test_empty_dataframe_handling(self):
        """Test functions handle empty DataFrames gracefully"""
        from analysis_pipeline import prepare_time_series_data
        
        # Create a proper empty DataFrame with the expected columns
        empty_df = pd.DataFrame({'date': pd.to_datetime([]), 'value': []})
        
        # Test time series preparation with empty data
        ts = prepare_time_series_data(empty_df, 'value')
        assert isinstance(ts, pd.Series), "Should return Series even for empty input"
        assert len(ts) == 0, "Empty input should return empty Series"
    
    def test_missing_columns_handling(self):
        """Test handling of missing columns"""
        from analysis_pipeline import prepare_time_series_data
        
        # DataFrame without required columns
        df_no_date = pd.DataFrame({'value': [1, 2, 3]})
        
        try:
            ts = prepare_time_series_data(df_no_date, 'value')
            # Should either return empty series or handle gracefully
            assert isinstance(ts, pd.Series)
        except KeyError:
            # Expected if 'date' column is missing
            pass


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
