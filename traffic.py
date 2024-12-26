import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from statsmodels.tsa.arima.model import ARIMA

traffic_df = pd.read_csv("")
location_df = pd.read_csv("")

traffic_df['timestamp'] = pd.to_datetime(traffic_df['timestamp'])

traffic_df['hour'] = traffic_df['timestamp'].dt.hour

peak_traffic = traffic_df.groupby('hour').agg({
    'vehicle_count': 'sum',
    'avg_speed': 'mean'
}).reset_index()

max_vehicle_count = peak_traffic['vehicle_count'].max()
peak_hours = peak_traffic[peak_traffic['vehicle_count'] == max_vehicle_count]

print("Peak Traffic Analysis:")
print(peak_traffic)
print("\nPeak Hours:")
print(peak_hours)

peak_traffic.to_csv("peak_traffic_analysis.csv", index=False)
print("\nAnalysis saved to 'peak_traffic_analysis.csv'")


# Task 2
merged_df = traffic_df.merge(location_df, on='location_id')

merged_df['date'] = merged_df['timestamp'].dt.date
merged_df['hour'] = merged_df['timestamp'].dt.hour

regional_trends = merged_df.groupby(['region', 'date']).agg({
    'vehicle_count': 'sum'
}).reset_index()

average_daily_traffic = regional_trends.groupby('region').agg({
    'vehicle_count': 'mean'
}).rename(columns={'vehicle_count': 'avg_daily_traffic'}).reset_index()

print("Regional Traffic Trends:")
print(regional_trends)
print("\nAverage Daily Traffic by Region:")
print(average_daily_traffic)

# Plot daily traffic trends for each region
plt.figure(figsize=(12, 6))
sns.lineplot(data=regional_trends, x='date', y='vehicle_count', hue='region', marker='o')
plt.title("Daily Traffic Trends by Region")
plt.xlabel("Date")
plt.ylabel("Vehicle Count")
plt.legend(title="Region")
plt.xticks(rotation=45)
plt.grid()
plt.tight_layout()
plt.savefig("daily_traffic_trends_by_region.png")
plt.show()

# Plot average daily traffic by region
plt.figure(figsize=(8, 5))
sns.barplot(data=average_daily_traffic, x='region', y='avg_daily_traffic', hue='region', palette='viridis', dodge=False, legend=False)
plt.title("Average Daily Traffic by Region")
plt.xlabel("Region")
plt.ylabel("Average Daily Traffic")
plt.grid(axis='y')
plt.tight_layout()
plt.savefig("average_daily_traffic_by_region.png")
plt.show()

regional_trends.to_csv("regional_traffic_trends.csv", index=False)
average_daily_traffic.to_csv("average_daily_traffic.csv", index=False)

print("\nAnalysis saved to 'regional_traffic_trends.csv' and 'average_daily_traffic.csv'")


# Task 3
weather_impact = traffic_df.groupby('weather_condition').agg({
    'vehicle_count': 'mean',
    'avg_speed': 'mean'
}).rename(columns={
    'vehicle_count': 'avg_vehicle_count',
    'avg_speed': 'avg_speed'
}).reset_index()

plt.figure(figsize=(12, 6))
sns.barplot(
    data=weather_impact.sort_values('avg_vehicle_count', ascending=False),
    x='weather_condition',
    y='avg_vehicle_count',
    hue='weather_condition',
    palette='coolwarm',
    dodge=False,
    legend=False
)

plt.title("Average Traffic Volume by Weather Condition")
plt.xlabel("Weather Condition")
plt.ylabel("Average Vehicle Count")
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.tight_layout()
plt.savefig("average_vehicle_count_by_weather.png")
plt.show()

plt.figure(figsize=(12, 6))
sns.barplot(
    data=weather_impact.sort_values('avg_speed', ascending=False),
    x='weather_condition',
    y='avg_speed',
    hue='weather_condition',
    palette='viridis',
    dodge=False,
    legend=False
)

plt.title("Average Traffic Speed by Weather Condition")
plt.xlabel("Weather Condition")
plt.ylabel("Average Speed (km/h)")
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.tight_layout()
plt.savefig("average_speed_by_weather.png")
plt.show()

weather_impact.to_csv("weather_impact_on_traffic.csv", index=False)

print("\nAnalysis saved to 'weather_impact_on_traffic.csv'")


# Task 4
merged_df = traffic_df.merge(location_df, on='location_id', how='inner')

merged_df.rename(columns={'road_type_x': 'traffic_road_type', 'road_type_y': 'location_road_type'}, inplace=True)

hotspot_analysis = merged_df.groupby(['location_id', 'location_road_type']).agg({
    'accident_count': 'sum'
}).reset_index()

top_hotspots = hotspot_analysis.nlargest(10, 'accident_count')

plt.figure(figsize=(12, 6))
sns.barplot(
    data=top_hotspots,
    x='location_id',
    y='accident_count',
    hue='location_road_type',
    dodge=False,
    palette='tab10'
)
plt.title("Top 10 Accident Hotspots")
plt.xlabel("Location ID")
plt.ylabel("Total Accidents")
plt.xticks(rotation=45)
plt.legend(title="Road Type")
plt.grid(axis='y')
plt.tight_layout()
plt.savefig("top_accident_hotspots.png")
plt.show()

road_type_analysis = merged_df.groupby('location_road_type').agg({
    'accident_count': 'sum'
}).reset_index()

plt.figure(figsize=(10, 6))
sns.barplot(
    data=road_type_analysis.sort_values('accident_count', ascending=False),
    x='location_road_type',
    y='accident_count',
    hue='location_road_type',
    palette='Set2',
    legend=False
)
plt.title("Accidents by Road Type")
plt.xlabel("Road Type")
plt.ylabel("Total Accidents")
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.tight_layout()
plt.savefig("accidents_by_road_type.png")
plt.show()

hotspot_analysis.to_csv("accident_hotspot_analysis.csv", index=False)

print("\nAnalysis saved to 'accident_hotspot_analysis.csv'")


# Task 5
traffic_df.set_index('timestamp', inplace=True)

daily_traffic = traffic_df.resample('D').sum()

plt.figure(figsize=(12, 6))
plt.plot(daily_traffic.index, daily_traffic['vehicle_count'], label='Historical Traffic Volume', color='blue')
plt.xlabel('Date')
plt.ylabel('Vehicle Count')
plt.title('Historical Traffic Volume (Daily)')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Split data into training and test sets
train_size = int(len(daily_traffic) * 0.8)
train, test = daily_traffic[:train_size], daily_traffic[train_size:]

model = ARIMA(train['vehicle_count'], order=(5, 1, 0))
model_fit = model.fit()

forecast = model_fit.forecast(steps=len(test))

plt.figure(figsize=(12, 6))
plt.plot(train.index, train['vehicle_count'], label='Train Data', color='blue')
plt.plot(test.index, test['vehicle_count'], label='Test Data (Actual)', color='orange')
plt.plot(test.index, forecast, label='Forecasted Data', color='green', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Vehicle Count')
plt.title('Traffic Volume Forecasting (ARIMA)')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

mae = mean_absolute_error(test['vehicle_count'], forecast)
print(f'Mean Absolute Error (MAE) on Test Set: {mae}')

future_forecast = model_fit.forecast(steps=30)

future_forecast = np.maximum(future_forecast, 0)
future_dates = pd.date_range(daily_traffic.index[-1], periods=31, freq='D')[1:]

plt.figure(figsize=(12, 6))
plt.plot(daily_traffic.index, daily_traffic['vehicle_count'], label='Historical Traffic Volume', color='blue')
plt.plot(future_dates, future_forecast, label='Future Forecast (Next 30 Days)', color='red', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Vehicle Count')
plt.title('Future Traffic Volume Forecast (Next 30 Days)')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Task 6
merged_df = pd.merge(traffic_df, location_df[['location_id', 'road_type']], on='location_id', how='inner')
merged_df.rename(columns={'road_type_x': 'traffic_road_type', 'road_type_y': 'location_road_type'}, inplace=True)
road_type_performance = merged_df.groupby('location_road_type').agg(
    total_vehicle_count=('vehicle_count', 'sum'),
    avg_vehicle_speed=('avg_speed', 'mean')
).reset_index()

plt.figure(figsize=(10, 6))
plt.bar(road_type_performance['location_road_type'], road_type_performance['total_vehicle_count'], color='blue')
plt.xlabel('Road Type')
plt.ylabel('Total Vehicle Count')
plt.title('Total Vehicle Count by Road Type')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.bar(road_type_performance['location_road_type'], road_type_performance['avg_vehicle_speed'], color='green')
plt.xlabel('Road Type')
plt.ylabel('Average Vehicle Speed (km/h)')
plt.title('Average Vehicle Speed by Road Type')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

fig, ax1 = plt.subplots(figsize=(12, 6))

ax1.set_xlabel('Road Type')
ax1.set_ylabel('Total Vehicle Count', color='blue')
ax1.bar(road_type_performance['location_road_type'], road_type_performance['total_vehicle_count'], color='blue', alpha=0.6)
ax1.tick_params(axis='y', labelcolor='blue')

ax2 = ax1.twinx()
ax2.set_ylabel('Average Vehicle Speed (km/h)', color='green')
ax2.plot(road_type_performance['location_road_type'], road_type_performance['avg_vehicle_speed'], color='green', marker='o', linestyle='dashed')
ax2.tick_params(axis='y', labelcolor='green')

fig.tight_layout()
plt.title('Road Type Performance: Total Vehicle Count vs. Average Speed')
plt.xticks(rotation=45)
plt.show()


# Task 7
merged_df = pd.merge(traffic_df, location_df[['location_id', 'region', 'road_type']], on='location_id', how='inner')
merged_df.rename(columns={'road_type_x': 'traffic_road_type', 'road_type_y': 'location_road_type'}, inplace=True)

region_road_type_performance = merged_df.groupby(['region', 'location_road_type']).agg(
    total_vehicle_count=('vehicle_count', 'sum'),
    avg_vehicle_speed=('avg_speed', 'mean')
).reset_index()

# Plot total vehicle count by region and road type
plt.figure(figsize=(12, 6))
for region in region_road_type_performance['region'].unique():
    region_data = region_road_type_performance[region_road_type_performance['region'] == region]
    plt.bar(region_data['location_road_type'], region_data['total_vehicle_count'], label=f'Region: {region}', alpha=0.6)
plt.xlabel('Road Type')
plt.ylabel('Total Vehicle Count')
plt.title('Total Vehicle Count by Region and Road Type')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plot average vehicle speed by region and road type
plt.figure(figsize=(12, 6))
for region in region_road_type_performance['region'].unique():
    region_data = region_road_type_performance[region_road_type_performance['region'] == region]
    plt.plot(region_data['location_road_type'], region_data['avg_vehicle_speed'], label=f'Region: {region}', marker='o')
plt.xlabel('Road Type')
plt.ylabel('Average Vehicle Speed (km/h)')
plt.title('Average Vehicle Speed by Region and Road Type')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Task 8
traffic_df = pd.read_csv("")

traffic_df['timestamp'] = pd.to_datetime(traffic_df['timestamp'])
traffic_df['year'] = traffic_df['timestamp'].dt.year
traffic_df['month'] = traffic_df['timestamp'].dt.month

monthly_traffic = traffic_df.groupby(['year', 'month'])['vehicle_count'].sum().reset_index()

monthly_traffic['month_year'] = monthly_traffic['year'].astype(str) + '-' + monthly_traffic['month'].astype(str).str.zfill(2)

monthly_traffic['month_year'] = pd.to_datetime(monthly_traffic['month_year'], format='%Y-%m')

plt.figure(figsize=(12, 6))
plt.plot(monthly_traffic['month_year'], monthly_traffic['vehicle_count'], marker='o', linestyle='-', color='b')
plt.xlabel('Month-Year')
plt.ylabel('Total Vehicle Count')
plt.title('Long-Term Traffic Volume Trends by Month')
plt.grid(True)
plt.xticks(rotation=45)
plt.show()

X = (monthly_traffic['month_year'] - monthly_traffic['month_year'].min()).dt.days.values.reshape(-1, 1)
y = monthly_traffic['vehicle_count'].values

model = LinearRegression()
model.fit(X, y)

# Make future predictions
future_months = pd.date_range(monthly_traffic['month_year'].max(), periods=13, freq='ME')  # 12 months forecast
future_months_numeric = (future_months - monthly_traffic['month_year'].min()).days.values.reshape(-1, 1)
predicted_vehicle_counts = model.predict(future_months_numeric)

plt.figure(figsize=(12, 6))
plt.plot(monthly_traffic['month_year'], monthly_traffic['vehicle_count'], marker='o', linestyle='-', color='b', label='Historical Data')
plt.plot(future_months, predicted_vehicle_counts, marker='x', linestyle='--', color='r', label='Predicted Data')
plt.xlabel('Month-Year')
plt.ylabel('Vehicle Count')
plt.title('Long-Term Traffic Volume Trend by Month with Forecasting')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.show()

future_predictions = pd.DataFrame({
    'month_year': future_months,
    'predicted_vehicle_count': predicted_vehicle_counts
})
print(future_predictions)


# Task 9
merged_df = pd.merge(traffic_df, location_df, on='location_id', how='inner')

merged_df['weather_condition'] = merged_df['weather_condition'].astype('category')

plt.figure(figsize=(12, 6))
sns.boxplot(x='weather_condition', y='vehicle_count', data=merged_df)
plt.title('Traffic Volume by Weather Condition')
plt.xlabel('Weather Condition')
plt.ylabel('Vehicle Count')
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(x='weather_condition', y='avg_speed', data=merged_df)
plt.title('Average Speed by Weather Condition')
plt.xlabel('Weather Condition')
plt.ylabel('Average Speed (km/h)')
plt.xticks(rotation=45)
plt.show()

# I will consider that high vehicle counts and low speeds could negatively affect air quality
merged_df['air_quality_impact'] = merged_df['vehicle_count'] * (1 / merged_df['avg_speed'])

sns.boxplot(x='weather_condition', y='air_quality_impact', data=merged_df)
plt.title('Impact of Traffic on Air Quality by Weather Condition')
plt.xlabel('Weather Condition')
plt.ylabel('Air Quality Impact (Vehicle Count / Avg Speed)')
plt.xticks(rotation=45)
plt.show()

correlation = merged_df[['vehicle_count', 'avg_speed', 'air_quality_impact']].corr()
print("Correlation matrix:\n", correlation)

plt.figure(figsize=(12, 6))
sns.scatterplot(x='vehicle_count', y='air_quality_impact', hue='weather_condition', data=merged_df)
plt.title('Vehicle Count vs. Air Quality Impact')
plt.xlabel('Vehicle Count')
plt.ylabel('Air Quality Impact (Vehicle Count / Avg Speed)')
plt.legend(title='Weather Condition')
plt.show()

plt.figure(figsize=(12, 6))
sns.scatterplot(x='avg_speed', y='air_quality_impact', hue='weather_condition', data=merged_df)
plt.title('Average Speed vs. Air Quality Impact')
plt.xlabel('Average Speed (km/h)')
plt.ylabel('Air Quality Impact (Vehicle Count / Avg Speed)')
plt.legend(title='Weather Condition')
plt.show()

label_encoder = LabelEncoder()
merged_df['weather_condition'] = label_encoder.fit_transform(merged_df['weather_condition'])

merged_df['air_quality_impact'] = merged_df['vehicle_count'] / merged_df['avg_speed']

features = merged_df[['vehicle_count', 'avg_speed', 'weather_condition']]
target = merged_df['air_quality_impact']

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Model Evaluation (Random Forest Regressor):")
print(f"Mean Absolute Error (MAE): {mean_absolute_error(y_test, y_pred)}")
print(f"Mean Squared Error (MSE): {mean_squared_error(y_test, y_pred)}")
print(f"R-squared (R2): {r2_score(y_test, y_pred)}")

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.title('Actual vs Predicted Air Quality Impact')
plt.xlabel('Actual Air Quality Impact')
plt.ylabel('Predicted Air Quality Impact')
plt.show()

feature_importances = model.feature_importances_
plt.barh(features.columns, feature_importances)
plt.title('Feature Importance in Predicting Air Quality Impact')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.show()


# Task 10
traffic_df['timestamp'] = pd.to_datetime(traffic_df['timestamp'])

merged_df = pd.merge(traffic_df, location_df, on='location_id', how='inner')

merged_df['hour_of_day'] = merged_df['timestamp'].dt.hour
merged_df['day_of_week'] = merged_df['timestamp'].dt.dayofweek
merged_df['is_weekend'] = merged_df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

# Encoding weather condition as numerical values (sunny=0, rainy=1, foggy=2)
weather_encoder = LabelEncoder()
merged_df['weather_condition_encoded'] = weather_encoder.fit_transform(merged_df['weather_condition'])

merged_df['lag_vehicle_count'] = merged_df['vehicle_count'].shift(1, fill_value=0)
merged_df['lag_avg_speed'] = merged_df['avg_speed'].shift(1, fill_value=merged_df['avg_speed'].mean())

merged_df.dropna(inplace=True)

# I predict vehicle count based on weather and traffic patterns
X = merged_df[['hour_of_day', 'day_of_week', 'is_weekend', 'weather_condition_encoded', 'lag_vehicle_count', 'lag_avg_speed']]
y = merged_df['vehicle_count']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = make_pipeline(StandardScaler(), RandomForestRegressor(n_estimators=100, random_state=42))

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"R² Score: {r2}")

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.6, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2)
plt.xlabel("Actual Vehicle Count")
plt.ylabel("Predicted Vehicle Count")
plt.title("Actual vs Predicted Vehicle Count")
plt.show()
