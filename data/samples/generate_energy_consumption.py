#!/usr/bin/env python3
"""
Generate sample energy consumption data with multiple seasonality patterns.
This creates realistic time series data that mimics electricity usage patterns.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os

def generate_energy_consumption(start_date='2018-01-01', periods=8760, freq='h'):
    """
    Generate synthetic hourly energy consumption data with the following components:
    - Daily seasonality (peak during daytime, low at night)
    - Weekly seasonality (lower on weekends)
    - Yearly seasonality (higher in winter and summer due to heating/cooling)
    - Gradual upward trend
    - Random noise
    - Weather impact simulation
    
    Parameters:
    -----------
    start_date : str
        Start date in 'YYYY-MM-DD' format
    periods : int
        Number of periods (hours) to generate
    freq : str
        Frequency of time series ('h' for hourly)
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with timestamp index and energy_consumption column
    """
    # Create timestamp range (hourly data)
    timestamp_rng = pd.date_range(start=start_date, periods=periods, freq=freq)
    
    # Initialize DataFrame
    df = pd.DataFrame(timestamp_rng, columns=['timestamp'])
    
    # Extract time features
    df['hour'] = df['timestamp'].dt.hour
    df['dayofweek'] = df['timestamp'].dt.dayofweek  # 0=Monday, 6=Sunday
    df['month'] = df['timestamp'].dt.month
    df['day'] = df['timestamp'].dt.day
    df['year'] = df['timestamp'].dt.year
    df['is_weekend'] = df['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)
    
    # Base load (minimum consumption)
    base_load = 20  # MW
    
    # Daily seasonality (hour of day pattern)
    # Night: low, morning: rising, afternoon/evening: peak, late evening: declining
    hourly_pattern = {
        0: 0.6, 1: 0.5, 2: 0.45, 3: 0.4, 4: 0.4, 5: 0.5,   # Night/early morning (0-5)
        6: 0.7, 7: 0.9, 8: 1.1, 9: 1.2, 10: 1.3, 11: 1.4,  # Morning/early afternoon (6-11)
        12: 1.5, 13: 1.5, 14: 1.5, 15: 1.4, 16: 1.3, 17: 1.4,  # Afternoon/evening (12-17)
        18: 1.5, 19: 1.6, 20: 1.5, 21: 1.3, 22: 1.0, 23: 0.8   # Evening/night (18-23)
    }
    df['hourly_pattern'] = df['hour'].map(hourly_pattern)
    
    # Weekly seasonality (weekend effect)
    df['weekday_factor'] = np.where(df['is_weekend'] == 1, 0.85, 1.0)
    
    # Monthly/seasonal pattern (heating/cooling needs)
    # Higher in winter (heating) and summer (cooling), lower in spring/fall
    monthly_pattern = {
        1: 1.4, 2: 1.3, 3: 1.1, 4: 0.9,    # Winter -> Spring
        5: 0.8, 6: 1.0, 7: 1.2, 8: 1.3,    # Spring -> Summer
        9: 1.0, 10: 0.9, 11: 1.1, 12: 1.3  # Fall -> Winter
    }
    df['monthly_pattern'] = df['month'].map(monthly_pattern)
    
    # Trend component (gradual increase over time)
    df['trend'] = df.index * 0.0001 + 1  # Small upward trend
    
    # Generate simulated temperature data (for weather effects)
    # Base annual temperature cycle
    df['day_of_year'] = df['timestamp'].dt.dayofyear
    df['temp_base'] = -np.cos(2 * np.pi * df['day_of_year'] / 365) * 15 + 15  # 0 to 30°C annual cycle
    
    # Add daily temperature cycle (warmer during day, cooler at night)
    df['temp_daily'] = -np.cos(2 * np.pi * df['hour'] / 24) * 5  # +/- 5°C daily cycle
    
    # Add random weather variations
    np.random.seed(42)  # for reproducibility
    temp_noise = np.random.normal(0, 2, len(df))  # Random weather variations
    
    # Combine temperature components
    df['temperature'] = df['temp_base'] + df['temp_daily'] + temp_noise
    
    # Calculate temperature impact on energy consumption
    # Higher consumption at temperature extremes (heating when cold, cooling when hot)
    df['temp_effect'] = 0.8 + 0.2 * np.abs(df['temperature'] - 20) / 20
    
    # Generate energy consumption with all components combined
    df['energy_consumption'] = (
        base_load * 
        df['hourly_pattern'] * 
        df['weekday_factor'] * 
        df['monthly_pattern'] * 
        df['trend'] * 
        df['temp_effect']
    )
    
    # Add random noise to represent unexpected variations
    noise = np.random.normal(1, 0.03, len(df))  # 3% noise
    df['energy_consumption'] = df['energy_consumption'] * noise
    
    # Round to two decimal places
    df['energy_consumption'] = df['energy_consumption'].round(2)
    
    # Add temperature to the final dataset and keep only relevant columns
    result = df[['timestamp', 'energy_consumption', 'temperature']].copy()
    
    return result

def main():
    """Generate and save sample energy consumption data."""
    # Create samples directory if it doesn't exist
    os.makedirs('data/samples', exist_ok=True)
    
    # Generate hourly energy consumption data for one year
    energy_data = generate_energy_consumption(
        start_date='2018-01-01', 
        periods=8760  # 24 hours * 365 days
    )
    
    # Save to CSV
    output_path = 'data/samples/energy_consumption_hourly.csv'
    energy_data.to_csv(output_path, index=False)
    print(f"Energy consumption data saved to {output_path}")
    
    # Create aggregated daily data for easier visualization
    daily_data = energy_data.copy()
    daily_data['date'] = daily_data['timestamp'].dt.date
    daily_agg = daily_data.groupby('date').agg({
        'energy_consumption': 'sum',
        'temperature': 'mean'
    }).reset_index()
    
    # Save daily aggregated data
    daily_output_path = 'data/samples/energy_consumption_daily.csv'
    daily_agg.to_csv(daily_output_path, index=False)
    print(f"Daily aggregated data saved to {daily_output_path}")
    
    # Generate preview plots
    fig, axes = plt.subplots(3, 1, figsize=(12, 15))
    
    # Plot 1: First two weeks of hourly data
    two_weeks = energy_data.iloc[:336]  # 24 hours * 14 days
    axes[0].plot(two_weeks['timestamp'], two_weeks['energy_consumption'])
    axes[0].set_title('Hourly Energy Consumption (First Two Weeks of 2018)')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Energy Consumption (MW)')
    axes[0].grid(True)
    
    # Plot 2: Full year of daily data
    axes[1].plot(daily_agg['date'], daily_agg['energy_consumption'])
    axes[1].set_title('Daily Energy Consumption (2018)')
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel('Energy Consumption (MW)')
    axes[1].grid(True)
    
    # Plot 3: Temperature vs Energy Consumption
    axes[2].scatter(energy_data['temperature'], energy_data['energy_consumption'], alpha=0.1)
    axes[2].set_title('Temperature vs. Energy Consumption')
    axes[2].set_xlabel('Temperature (°C)')
    axes[2].set_ylabel('Energy Consumption (MW)')
    axes[2].grid(True)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = 'data/samples/energy_consumption_preview.png'
    plt.savefig(plot_path)
    print(f"Preview plots saved to {plot_path}")
    
    # Display sample data
    print("\nSample hourly data (first 24 hours):")
    print(energy_data.head(24))
    
    print("\nData statistics:")
    print(energy_data['energy_consumption'].describe())

if __name__ == "__main__":
    main()