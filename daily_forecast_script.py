import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import streamlit as st
import time
from prophet import Prophet
import plotly.graph_objects as go
import pandas as pd
import plotly.io as pio
import plotly.express as px
from plotly.subplots import make_subplots
import os
from flask import Flask, render_template_string
import plotly.graph_objects as go
import matplotlib.gridspec as gridspec

start_time = time.time()

# from xgboost import XGBRegressor
# from sklearn.ensemble import RandomForestRegressor
# from catboost import CatBoostRegressor
def resampled_data(data, resample):
    daily_data = data.groupby(['item_id']).resample(resample).agg({
        'quantity': 'sum',
        'price': 'mean'
    }).reset_index().fillna(method='ffill')
    return daily_data

def data_proprocessing(data):
    data['day_of_month'] = data['transaction_date'].dt.day
    data['day_of_week'] = data['transaction_date'].dt.dayofweek
    data['month'] = data['transaction_date'].dt.month
    data['year'] = data['transaction_date'].dt.year

    data['lag_1'] = data['quantity'].shift(1)
    data['lag_7'] = data['quantity'].shift(7)

    data['rolling_mean_3'] = data['quantity'].rolling(window=3).mean()
    data['rolling_std_3'] = data['quantity'].rolling(window=3).std()
    data['rolling_mean_7'] = data['quantity'].rolling(window=7).mean()
    data['rolling_std_7'] = data['quantity'].rolling(window=7).std()

    data.dropna(inplace=True)
    data.drop('price', axis =1,  inplace = True)


    return data

def model_building(train_data, ws=False, ys=False, ds=False):
    model = Prophet(weekly_seasonality=ws, yearly_seasonality=ys, daily_seasonality=ds)
    model.add_regressor('day_of_week')
    model.add_regressor('day_of_month')
    model.add_regressor('year')
    model.add_regressor('month')
    model.add_regressor('lag_1')
    model.add_regressor('lag_7')
    model.add_regressor('rolling_std_3')
    model.add_regressor('rolling_mean_3')
    model.add_regressor('rolling_mean_7')
    model.add_regressor('rolling_std_7')
    model.fit(train_data)
    return model

def future_prediction(model, train_data, test_data, freq):
    future = model.make_future_dataframe(periods=len(test_data), freq=freq)
    future = future.loc[len(train_data):]
    future['day_of_week'] = test_data['day_of_week'].values
    future['day_of_month'] = test_data['day_of_month'].values
    future['month'] = test_data['month'].values
    future['year'] = test_data['year'].values
    future['lag_1'] = test_data['lag_1'].values
    future['lag_7'] = test_data['lag_7'].values 

    future['rolling_mean_3'] = test_data.rolling_mean_3.values
    future['rolling_std_3'] = test_data.rolling_std_3.values

    future['rolling_mean_7'] = test_data.rolling_mean_7.values
    future['rolling_std_7'] = test_data.rolling_std_7.values
    forecast = model.predict(future)
    forecast['yhat'] = forecast['yhat'].apply(lambda x: max(0, x))
    return forecast


def model_score(test_data, forecast):  
    mse = mean_squared_error(test_data['y'], forecast['yhat'])
    r2 = r2_score(test_data['y'], forecast['yhat'])
    rmse = np.sqrt(mse)
    print(f'Prophet RMSE: {rmse} \n R2: {r2}')
    return rmse, r2
    
def plot(model, train_data, test_data, forecast):
    fig = model.plot(forecast)
    plt.plot(train_data['ds'], train_data['y'], color = 'blue')
    plt.plot(test_data['ds'], test_data['y'], 'blue', label='Actual')
    plt.legend()
    plt.show()


def flatten_data(item_data):
    flattened_data = []


    for item in item_data.values():
        item_id = item['item_id']
        dates = item['date']
        actual_data = item['actual_data']
        forecast = item['forecast']

        item_records = list(zip([item_id]*len(dates), dates, actual_data, forecast))

        flattened_data.extend(item_records)
    df = pd.DataFrame(flattened_data, columns=['item_id', 'date', 'test_data', 'forecast'])
    return df

# def plot_all(df, time):
#     for item in df.item_id.unique():
#         item_df = df[df['item_id']== item]
#         plt.plot(item_df['date'], item_df['test_data'], color = 'blue', label = "Actual")
#         plt.plot(item_df['date'], item_df['forecast'], 'red', label='Forecast')
#         plt.legend()
#         if time =='D':
#             plt.title(f'ITEM: {item} Daily Forecast')
#         elif time == 'W':
#             plt.title(f'ITEM: {item} Weekly Forecast')
#         else:
#             plt.title(f'ITEM: {item} Monthly Forecast')
#         plt.xlabel("Year")
#         plt.ylabel("Quantity")
#         plt.show()

# def plot_all(daily_df, weekly_df, monthly_df, rows_per_page=20):
#     """
#     Plots daily, weekly, and monthly forecasts and data for each item with table pagination.

#     Args:
#         daily_df (pd.DataFrame): DataFrame containing daily data.
#         weekly_df (pd.DataFrame): DataFrame containing weekly data.
#         monthly_df (pd.DataFrame): DataFrame containing monthly data.
#         rows_per_page (int): Number of rows to display per page in tables.
#     """

#     unique_items = daily_df['item_id'].unique()

#     for item in unique_items:
#         item_daily_df = daily_df[daily_df['item_id'] == item]
#         item_weekly_df = weekly_df[weekly_df['item_id'] == item]
#         item_monthly_df = monthly_df[monthly_df['item_id'] == item]

#         # Calculate number of pages for each table
#         num_pages_daily = np.ceil(len(item_daily_df) / rows_per_page)
#         num_pages_weekly = np.ceil(len(item_weekly_df) / rows_per_page)
#         num_pages_monthly = np.ceil(len(item_monthly_df) / rows_per_page)

#         # Find the maximum number of pages across all tables
#         max_pages = max(num_pages_daily, num_pages_weekly, num_pages_monthly)

#         for page in range(int(max_pages)):
#             fig, axs = plt.subplots(2, 3, figsize=(18, 12))
#             fig.suptitle(f"Forecast and Data for ITEM: {item} - Page {page + 1}", fontsize=16)

#             # Daily plot
#             axs[0, 0].plot(item_daily_df['date'], item_daily_df['test_data'], color='blue', label='Actual')
#             axs[0, 0].plot(item_daily_df['date'], item_daily_df['forecast'], color='red', label='Forecast')
#             axs[0, 0].set_title("Daily Forecast")
#             axs[0, 0].set_xlabel("Date")
#             axs[0, 0].set_ylabel("Quantity")
#             axs[0, 0].legend()
#             plt.xticks(rotation=45)

#             # Weekly plot
#             axs[0, 1].plot(item_weekly_df['date'], item_weekly_df['test_data'], color='blue', label='Actual')
#             axs[0, 1].plot(item_weekly_df['date'], item_weekly_df['forecast'], color='red', label='Forecast')
#             axs[0, 1].set_title("Weekly Forecast")
#             axs[0, 1].set_xlabel("Date")
#             axs[0, 1].set_ylabel("Quantity")
#             axs[0, 1].legend()
#             plt.xticks(rotation=45)

#             # Monthly plot
#             axs[0, 2].plot(item_monthly_df['date'], item_monthly_df['test_data'], color='blue', label='Actual')
#             axs[0, 2].plot(item_monthly_df['date'], item_monthly_df['forecast'], color='red', label='Forecast')
#             axs[0, 2].set_title("Monthly Forecast")
#             axs[0, 2].set_xlabel("Date")
#             axs[0, 2].set_ylabel("Quantity")
#             axs[0, 2].legend()
#             plt.xticks(rotation=45)

#             # Daily table
#             start_row = page * rows_per_page
#             end_row = min(start_row + rows_per_page, len(item_daily_df))
#             axs[1, 0].axis('off')
#             table = axs[1, 0].table(cellText=item_daily_df.iloc[start_row:end_row].values,
#                                    colLabels=item_daily_df.columns,
#                                    cellLoc='center',
#                                    loc='center')
#             table.auto_set_font_size(False)
#             table.set_fontsize(8)
#             table.scale(1, 1.2)
#             axs[1, 0].set_title(f"Daily Data - Page {page + 1}")

#             # Weekly table
#             start_row = page * rows_per_page
#             end_row = min(start_row + rows_per_page, len(item_weekly_df))
#             axs[1, 1].axis('off')
#             table = axs[1, 1].table(cellText=item_weekly_df.iloc[start_row:end_row].values,
#                                    colLabels=item_weekly_df.columns,
#                                    cellLoc='center',
#                                    loc='center')
#             table.auto_set_font_size(False)
#             table.set_fontsize(8)
#             table.scale(1, 1.2)
#             axs[1, 1].set_title(f"Weekly Data - Page {page + 1}")

#             # Monthly table
#             start_row = page * rows_per_page
#             end_row = min(start_row + rows_per_page, len(item_monthly_df))
#             axs[1, 2].axis('off')
#             table = axs[1, 2].table(cellText=item_monthly_df.iloc[start_row:end_row].values,
#                                    colLabels=item_monthly_df.columns,
#                                    cellLoc='center',
#                                    loc='center')
#             table.auto_set_font_size(False)
#             table.set_fontsize(8)
#             table.scale(1, 1.2)
#             axs[1, 2].set_title(f"Monthly Data - Page {page + 1}")

#             plt.tight_layout()
#             plt.show()




def plot_all(daily_df, weekly_df, monthly_df, selected_item):
    fig = make_subplots(
        rows=2, cols=3,
        specs=[[{'type': 'scatter'}, {'type': 'scatter'}, {'type': 'scatter'}],
               [{'type': 'table'}, {'type': 'table'}, {'type': 'table'}]],
        subplot_titles=(f"ITEM: {selected_item} Daily Forecast", f"ITEM: {selected_item} Weekly Forecast", f"ITEM: {selected_item} Monthly Forecast",
                        f"Daily Data for ITEM: {selected_item}", f"Weekly Data for ITEM: {selected_item}", f"Monthly Data for ITEM: {selected_item}")
    )

    # Daily plot
    item_daily_df = daily_df[daily_df['item_id'] == selected_item]
    fig.add_trace(go.Scatter(x=item_daily_df['date'], y=item_daily_df['test_data'], mode='lines', name='Actual', line=dict(color='blue')), row=1, col=1)
    fig.add_trace(go.Scatter(x=item_daily_df['date'], y=item_daily_df['forecast'], mode='lines', name='Forecast', line=dict(color='red')), row=1, col=1)

    # Weekly plot
    item_weekly_df = weekly_df[weekly_df['item_id'] == selected_item]
    fig.add_trace(go.Scatter(x=item_weekly_df['date'], y=item_weekly_df['test_data'], mode='lines', name='Actual', line=dict(color='blue')), row=1, col=2)
    fig.add_trace(go.Scatter(x=item_weekly_df['date'], y=item_weekly_df['forecast'], mode='lines', name='Forecast', line=dict(color='red')), row=1, col=2)

    # Monthly plot
    item_monthly_df = monthly_df[monthly_df['item_id'] == selected_item]
    fig.add_trace(go.Scatter(x=item_monthly_df['date'], y=item_monthly_df['test_data'], mode='lines', name='Actual', line=dict(color='blue')), row=1, col=3)
    fig.add_trace(go.Scatter(x=item_monthly_df['date'], y=item_monthly_df['forecast'], mode='lines', name='Forecast', line=dict(color='red')), row=1, col=3)

    # Daily table
    fig.add_trace(go.Table(
        header=dict(values=list(item_daily_df.columns), fill_color='paleturquoise', align='left'),
        cells=dict(values=[item_daily_df[col] for col in item_daily_df.columns], fill_color='lavender', align='left')
    ), row=2, col=1)

    # Weekly table
    fig.add_trace(go.Table(
        header=dict(values=list(item_weekly_df.columns), fill_color='paleturquoise', align='left'),
        cells=dict(values=[item_weekly_df[col] for col in item_weekly_df.columns], fill_color='lavender', align='left')
    ), row=2, col=2)

    # Monthly table
    fig.add_trace(go.Table(
        header=dict(values=list(item_monthly_df.columns), fill_color='paleturquoise', align='left'),
        cells=dict(values=[item_monthly_df[col] for col in item_monthly_df.columns], fill_color='lavender', align='left')
    ), row=2, col=3)

    # Update layout for better spacing
    fig.update_layout(height=800, showlegend=False, title_text=f"Forecast and Data for ITEM: {selected_item}")
    fig.update_xaxes(tickangle=45)  # Rotate x-axis labels for all subplots

    return fig
        


if __name__ =='__main__':
    data = pd.read_csv('transaction_data_with_seasonality_100k.csv', parse_dates=['transaction_date']) 
    data.set_index('transaction_date', inplace=True)
    daily_data = resampled_data(data, 'D')
    weekly_data = resampled_data(data, 'W')
    monthly_data = resampled_data(data,'M')

    daily_preprocessed_data = data_proprocessing(daily_data)
    weekly_preprocessed_data = data_proprocessing(weekly_data)
    monthly_preprocessed_data = data_proprocessing(monthly_data)

    daily_item_data_prophet = daily_preprocessed_data.rename(columns={'transaction_date': 'ds', 'quantity': 'y'})
    weekly_item_data_prophet = weekly_preprocessed_data.rename(columns={'transaction_date': 'ds', 'quantity': 'y'})
    monthly_item_data_prophet = monthly_preprocessed_data.rename(columns={'transaction_date': 'ds', 'quantity': 'y'})
    results = {}
    daily_item_data = {}
    weekly_item_data = {}
    monthly_item_data = {}
    # daily 
    for item in daily_item_data_prophet['item_id'].unique():
        item_df = daily_item_data_prophet[(daily_item_data_prophet['item_id'] == item)].drop('item_id', axis = 1)
        # item_df.set_index('ds', inplace=True) 
        # Split the data into train and test sets

        print(item_df.columns)

        train_size = int(len(item_df) * 0.8) 
        train_data, test_data = item_df[:train_size], item_df[train_size:]  
        model = model_building(train_data, ws=True, ys=True, ds = True)
    
        forecast = future_prediction(model, train_data, test_data, 'D')
        rmse, r2 = model_score(test_data, forecast)
        results[f'daily forecast - {item}'] = {"daily_rmse": rmse, "daily_r2":r2}
        # combined_data = pd.concat([train_data[['ds', 'y']], test_data[['ds', 'y']]], axis=0)
        # print(combined_data.tail())
        daily_item_data[item] = {
            "item_id": item,
            "date": test_data['ds'],
            "actual_data": test_data['y'],
            # "test_data": test_data['y'],
            "forecast": forecast['yhat']
        }
        # plot(model, train_data, test_data, forecast)

    # weekly
    for item in weekly_item_data_prophet['item_id'].unique():
        item_df = weekly_item_data_prophet[(weekly_item_data_prophet['item_id'] == item)].drop('item_id', axis = 1)
        # item_df.set_index('ds', inplace=True) 
        # Split the data into train and test sets

        print(item_df.columns)

        train_size = int(len(item_df) * 0.8) 
        train_data, test_data = item_df[:train_size], item_df[train_size:]  
        model = model_building(train_data, ws=True, ys=True, ds = True)
    
        forecast = future_prediction(model, train_data, test_data, 'W')
        rmse, r2 = model_score(test_data, forecast)
        results[f'weekly forecast - {item}'] = {"weekly_rmse": rmse, "weekly_r2":r2}
        # combined_data = pd.concat([train_data[['ds', 'y']], test_data[['ds', 'y']]], axis=0)
        # print(combined_data.tail())
        weekly_item_data[item] = {
            "item_id": item,
            "date": test_data['ds'],
            "actual_data": test_data['y'],
            # "test_data": test_data['y'],
            "forecast": forecast['yhat']
        }
        # plot(model, train_data, test_data, forecast)

    # monthly
    for item in monthly_item_data_prophet['item_id'].unique():
        item_df = monthly_item_data_prophet[(monthly_item_data_prophet['item_id'] == item)].drop('item_id', axis = 1)
        # item_df.set_index('ds', inplace=True) 
        # Split the data into train and test sets

        print(item_df.columns)

        train_size = int(len(item_df) * 0.8) 
        train_data, test_data = item_df[:train_size], item_df[train_size:]  
        model = model_building(train_data, ws=False, ys=True, ds = False)
    
        forecast = future_prediction(model, train_data, test_data, 'M')
        rmse, r2 = model_score(test_data, forecast)
        results[f'monthly forecast - {item}'] = {"monthly_rmse": rmse, "monthly_r2":r2}
        # combined_data = pd.concat([train_data[['ds', 'y']], test_data[['ds', 'y']]], axis=0)
        # print(combined_data.tail())
        monthly_item_data[item] = {
            "item_id": item,
            "date": test_data['ds'],
            "actual_data": test_data['y'],
            # "test_data": test_data['y'],
            "forecast": forecast['yhat']
        }
        # plot(model, train_data, test_data, forecast)
    
    daily_flatten_data = flatten_data(daily_item_data)
    print(daily_flatten_data.columns)
    weekly_flatten_data = flatten_data(weekly_item_data)
    monthly_flatten_data = flatten_data(monthly_item_data)
    # plot_all(daily_flatten_data, weekly_flatten_data, monthly_flatten_data)
    # plot_all(weekly_flatten_data, 'W')
    # plot_all(monthly_flatten_data, 'M')

    # daily_flatten_data.to_csv('Daily Forecast.csv', index = False)
    # weekly_flatten_data.to_csv('Weekly Forecast.csv', index= False)
    # monthly_flatten_data.to_csv('Monthly Forecast.csv', index = False)
    print(results)
    # End the timer
    end_time = time.time()

    st.title('Item Forecasting Dashboard')

    # Item selection
    unique_items = daily_flatten_data['item_id'].unique()
    selected_item = st.selectbox('Select an Item ID:', unique_items)

    # Plot data
    fig = plot_all(daily_flatten_data, weekly_flatten_data, monthly_flatten_data, selected_item)
    st.plotly_chart(fig)
    # Calculate the elapsed time
    runtime = end_time - start_time

    # Print the runtime
    print(f"Script runtime: {runtime} seconds")







        

        