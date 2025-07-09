import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import os

print("Loading CSV file...")
df = pd.read_csv("Cleaned_Superstore_Sales.xlsx.csv", encoding='latin1')
print("File loaded!")

print("Converting Order Date to datetime...")
df['Order Date'] = pd.to_datetime(df['Order Date'], dayfirst=True)

print("Grouping by Month and summing Sales...")
df['Month'] = df['Order Date'].dt.to_period('M')
monthly_sales = df.groupby('Month')['Sales'].sum().reset_index()
monthly_sales['Month'] = monthly_sales['Month'].dt.to_timestamp()

print("Preparing data for Prophet...")
df_prophet = monthly_sales.rename(columns={'Month': 'ds', 'Sales': 'y'})

print("Training Prophet model...")
model = Prophet()
model.fit(df_prophet)

print("Forecasting next 6 months...")
future = model.make_future_dataframe(periods=6, freq='M')
forecast = model.predict(future)

print("Plotting forecast chart...")
model.plot(forecast)
plt.title("Sales Forecast")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.tight_layout()
plt.show()

# Save to Desktop
print("Saving forecast to Desktop...")
forecast_result = forecast[['ds', 'yhat']].rename(columns={'ds': 'Month', 'yhat': 'Predicted Sales'})
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop", "Forecasted_Sales.csv")
forecast_result.to_csv(desktop_path, index=False)

print("\n Forecasted_Sales.csv has been saved to your Desktop!")
