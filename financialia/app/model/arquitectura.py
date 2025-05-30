import numpy as np
import pandas as pd
import datetime
import yfinance as yf
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from math import sqrt
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
import joblib
import os
import json


class Model_LSTM:
    def __init__(self, variables, target_variable, start_date, look_back=30, future_periods=3):
        self.variables = variables
        self.target_variable = target_variable
        self.start_date = start_date
        self.end_date = datetime.date.today().strftime('%Y-%m-%d')
        self.look_back = look_back
        self.future_periods = future_periods
        self.model = None
        self.scaler = None
        self.data = None
        self.data_returns = None
        self.data_scaled = None
        self.dates = None
       
    def load_data(self):
        data = yf.download(self.variables, start=self.start_date, end=self.end_date)['Close']
        self.data = data
        return data
   
    def preprocess_data(self, data=None):
        if data is None:
            data = self.data
            
        if data is None:
            raise ValueError("No data available. Please load data first.")
            
        imputer = KNNImputer(n_neighbors=2, weights="distance")
        data_imputed = imputer.fit_transform(data)
        data = pd.DataFrame(data_imputed, columns=data.columns, index=data.index)
        
        data_returns = data.pct_change().shift(-1).dropna()
        self.data_returns = data_returns
        
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        data_scaled = self.scaler.fit_transform(data_returns)
        self.data_scaled = data_scaled
        self.dates = data_returns.index
        
        return data_scaled, data_returns.index, data_returns      
    
    def create_dataset(self, dataset=None):
        if dataset is None:
            dataset = self.data_scaled
            
        target_index = self.variables.index(self.target_variable)
        data_x, data_y = [], []
        for i in range(self.look_back, len(dataset)):
            data_x.append(dataset[i - self.look_back:i, :])
            data_y.append(dataset[i, target_index])
            
        return np.array(data_x), np.array(data_y)  
    
    def split_data(self, X, y, train_ratio=0.8):
        train_size = int(len(X) * train_ratio)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
        return X_train, X_test, y_train, y_test
    
    def build_model(self, lstm_units=180, dropout_rate=0.1014, dense_units=100, learning_rate=0.001025, activation='tanh'):
        model = Sequential()
        model.add(LSTM(units=lstm_units, activation=activation, return_sequences=True,
                       dropout=dropout_rate, recurrent_dropout=dropout_rate,
                       input_shape=(self.look_back, len(self.variables))))
        model.add(LSTM(units=lstm_units // 2, activation=activation,
                       dropout=dropout_rate, recurrent_dropout=dropout_rate))
        model.add(Dense(dense_units, activation=activation))
        model.add(Dense(1))

        optimizer = tf.keras.optimizers.AdamW(learning_rate=learning_rate)
        model.compile(loss='mean_squared_error', optimizer=optimizer)

        self.model = model
        return model
    
    def train_model(self, X_train, y_train, X_val, y_val, epochs=200, batch_size=32):
        callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=200, restore_best_weights=True)]
        history = self.model.fit(X_train, y_train,
                                 validation_data=(X_val, y_val),
                                 epochs=epochs, batch_size=batch_size,
                                 callbacks=callbacks, verbose=2)
        return history
    
    def evaluate_model(self, X_test, y_test):
        predictions = self.model.predict(X_test)
        target_index = self.variables.index(self.target_variable)

        zeros_array = np.zeros((len(predictions), len(self.variables)))

        pred_array = zeros_array.copy()
        pred_array[:, target_index] = predictions.flatten()
        
        actual_array = zeros_array.copy()
        actual_array[:, target_index] = y_test.flatten()
       
        predictions_inverse = self.scaler.inverse_transform(pred_array)[:, target_index]
        y_test_inverse = self.scaler.inverse_transform(actual_array)[:, target_index]

        # Calculate metrics
        rmse = sqrt(mean_squared_error(y_test_inverse, predictions_inverse))
        mae = mean_absolute_error(y_test_inverse, predictions_inverse)
        mape = mean_absolute_percentage_error(y_test_inverse, predictions_inverse)
        
        print(f"Test RMSE: {rmse:.4f}")
        print(f"Test MAE: {mae:.4f}")
        print(f"Test MAPE: {mape:.4f}")
        
        return predictions_inverse, y_test_inverse, rmse, mae, mape
    
  
    def save_model(self, model_dir='modelo'):
        """Save the model and scaler"""
        try:
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            
            model_name = f"{self.target_variable}_model"
            
            model_info = {
                'variables': self.variables,
                'target_variable': self.target_variable,
                'look_back': self.look_back,
                'future_periods': self.future_periods,
                'start_date': self.start_date,
                'end_date': self.end_date
            }
           
            json_path = os.path.join(model_dir, f"{model_name}_info.json")
            model_path = os.path.join(model_dir, f"{model_name}.h5")
            scaler_path = os.path.join(model_dir, f"{model_name}_scaler.pkl")
           
            with open(json_path, 'w') as f:
                json.dump(model_info, f)
             
            self.model.save(model_path)
            
            joblib.dump(self.scaler, scaler_path)
            
            return model_name
        except Exception as e:
            print(f"Error al guardar el modelo: {str(e)}")
            raise
    
    def predict_future(self, periods=None):
        if periods is None:
            periods = self.future_periods

        if self.model is None:
            raise ValueError("Model hasn't been trained yet.")

        if self.data is None:
            raise ValueError("No data available. Please load data first.")

        latest_data = self.data.iloc[-self.look_back:].copy()

        latest_returns = latest_data.pct_change().shift(-1).dropna()

        latest_scaled = self.scaler.transform(latest_returns)

        latest_scaled.reshape(1, latest_scaled.shape[0], latest_scaled.shape[1])

        target_index = self.variables.index(self.target_variable)

        predictions = []
        future_dates = []

        last_date = self.data.index[-1]

        current_sequence = latest_scaled.copy()

        for i in range(periods):
          
            pred = self.model.predict(current_sequence.reshape(1, self.look_back-1, len(self.variables)))

            zeros_array = np.zeros((1, len(self.variables)))
            pred_array = zeros_array.copy()
            pred_array[0, target_index] = pred[0, 0]

            pred_return = self.scaler.inverse_transform(pred_array)[0, target_index]

            last_price = self.data[self.target_variable].iloc[-1] if i == 0 else predictions[-1]['price']
            future_price = last_price * (1 + pred_return)

            if i == 0:
                future_date = pd.to_datetime(last_date) + pd.Timedelta(days=1)
            else:
                future_date = pd.to_datetime(future_dates[-1]) + pd.Timedelta(days=1)

            while future_date.weekday() > 4: 
                future_date += pd.Timedelta(days=1)

            future_dates.append(future_date)

            predictions.append({
                'date': future_date.strftime('%Y-%m-%d'),
                'target_variable': self.target_variable,
                'return': float(pred_return),
                'price': float(future_price)
            })

            new_row = np.zeros((1, len(self.variables)))
            new_row[0, target_index] = pred[0, 0]
            current_sequence = np.vstack((current_sequence[1:], new_row))

        return predictions

    
    def get_available_models(model_dir='modelo'):
        """Get list of available saved models"""
        if not os.path.exists(model_dir):
            return []
            
        model_files = [f for f in os.listdir(model_dir) if f.endswith('_info.json')]
        models = []
        
        for model_file in model_files:
            model_name = model_file.replace('_info.json', '')
            
            # Load model info
            with open(f"{model_dir}/{model_file}", 'r') as f:
                model_info = json.load(f)
                
            models.append({
                'name': model_name,
                'target': model_info['target_variable'],
                'variables': model_info['variables'],
                'date_created': model_name.split('_')[-2] + '_' + model_name.split('_')[-1]
            })
            
        return models
    
