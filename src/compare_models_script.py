import tensorflow as tf

# Check the available physical devices
print("Available GPUs: ", tf.config.list_physical_devices('GPU'))

# Optionally, set memory growth to avoid excessive GPU memory usage
gpu_devices = tf.config.list_physical_devices('GPU')
if gpu_devices:
    for gpu in gpu_devices:
        tf.config.experimental.set_memory_growth(gpu, True)


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU, Flatten, Input, Dropout
from tensorflow.keras.optimizers import Adam
import yfinance as yf

# Prompt user for inputs
ticker = input("Enter the stock ticker symbol (e.g., AAPL): ")
start_date = input("Enter the start date (YYYY-MM-DD): ")
end_date = input("Enter the end date (YYYY-MM-DD): ")

num_lags = int(input("Enter the number of lags for the models: "))
number_of_hidden = int(input("Enter the number of hidden layers (>=2): "))
neurons = int(input("Enter the number of neurons for hidden layers: "))

epochs = int(input("Enter the number of epochs for training: "))
learning_rate = float(input("Enter the learning rate: "))
batch_size = int(input("Enter the batch size for training: "))
dropout_rate = float(input("Enter the dropout rate for NNs to avoid overfitting (typically from 0.2 to 0.5): "))

# make folder for output plots
out_dir = f"../experiments/lags{num_lags}_hidden{number_of_hidden}_neurons{neurons}_epochs{epochs}_lr{learning_rate}_bs{batch_size}_dropout{dropout_rate}"
os.makedirs(out_dir, exist_ok=True)  # Create the directory if it doesn't exist

# Fetch stock data
data = yf.download(ticker, start=start_date, end=end_date)
data_pd_series = data['Close'].squeeze()

# Linear Regression
df_lr = pd.DataFrame({'y': data_pd_series})

for lag in range(1, num_lags + 1):
    df_lr[f'y_lag_{lag}'] = df_lr['y'].shift(lag)

df_lr = df_lr.dropna()

X_lr = df_lr[[f'y_lag_{lag}' for lag in range(1, num_lags + 1)]]
y_lr = df_lr['y']
y_lr, X_lr = y_lr.align(X_lr, join='inner')

num_test_samples = 10

X_lr_train, X_lr_test = X_lr.iloc[:-num_test_samples], X_lr.iloc[-num_test_samples:]
y_lr_train, y_lr_test = y_lr.iloc[:-num_test_samples], y_lr.iloc[-num_test_samples:]

model = LinearRegression()
model.fit(X_lr_train, y_lr_train)

lr_train_preds = pd.Series(model.predict(X_lr_train), index=X_lr_train.index)
lr_pred_list = []
lr_lag_values = X_lr_test.iloc[0, 1:].values

for i in range(len(X_lr_test)):
    X_lr_test_row = np.concatenate(([X_lr_test.iloc[i, 0]], lr_lag_values))
    X_lr_test_row_df = pd.DataFrame([X_lr_test_row], columns=X_lr_train.columns)
    lr_pred = model.predict(X_lr_test_row_df)[0]
    lr_pred_list.append(lr_pred)
    lr_lag_values = np.roll(lr_lag_values, -1)
    lr_lag_values[-1] = lr_pred

lr_preds = pd.Series(lr_pred_list, index=X_lr_test.index)

# Neural Networks
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data[['Close']])
def create_sequences(data, seq_length):
    sequences, targets = [], []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i+seq_length])
        targets.append(data[i+seq_length])
    return np.array(sequences), np.array(targets)
seq_length = num_lags
X, y = create_sequences(data_scaled, seq_length)
train_size = int(len(X) - 10)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# LSTM Model
lstm_model = Sequential([
    Input(shape=(seq_length, 1)),
    #*[LSTM(neurons, return_sequences=True) for _ in range(number_of_hidden-1)],
    *[layer for _ in range(number_of_hidden-1) 
        for layer in (LSTM(neurons, return_sequences=True), Dropout(dropout_rate))
        ],
    LSTM(neurons, return_sequences=False),
    Dense(1)
])

with tf.device('/GPU:0'):  # Force LSTM to run on GPU
    lstm_model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
    lstm_model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
    lstm_train_preds = lstm_model.predict(X_train)
    lstm_rolling_preds = []
    lstm_current_sequence = X_test[0].reshape(1, seq_length, 1)
    for _ in range(len(X_test)):
        next_pred = lstm_model.predict(lstm_current_sequence, verbose=0)[0, 0]
        lstm_rolling_preds.append(next_pred)
        lstm_current_sequence = np.roll(lstm_current_sequence, -1, axis=1)
        lstm_current_sequence[0, -1, 0] = next_pred

lstm_test_preds = scaler.inverse_transform(np.array(lstm_rolling_preds).reshape(-1, 1))
lstm_train_preds = scaler.inverse_transform(lstm_train_preds)

# GRU Model
gru_model = Sequential([
    Input(shape=(seq_length, 1)),
    #*[GRU(neurons, return_sequences=True) for _ in range(number_of_hidden-1)],  # Loop inside the list
    *[layer for _ in range(number_of_hidden-1) 
        for layer in (GRU(neurons, return_sequences=True), Dropout(dropout_rate))
        ],
    GRU(neurons, return_sequences=False),
    Dense(1)
])

with tf.device('/GPU:0'):  # Force LSTM to run on GPU
    gru_model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
    gru_model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
    gru_train_preds = gru_model.predict(X_train)
    gru_rolling_preds = []
    gru_current_sequence = X_test[0].reshape(1, seq_length, 1)
    for _ in range(len(X_test)):
        next_pred = gru_model.predict(gru_current_sequence, verbose=0)[0, 0]
        gru_rolling_preds.append(next_pred)
        gru_current_sequence = np.roll(gru_current_sequence, -1, axis=1)
        gru_current_sequence[0, -1, 0] = next_pred

gru_test_preds = scaler.inverse_transform(np.array(gru_rolling_preds).reshape(-1, 1))
gru_train_preds = scaler.inverse_transform(gru_train_preds)

# Simple NN Model
nn_model = Sequential([
    Input(shape=(seq_length, 1)),
    Flatten(),
    #*[Dense(neurons, activation='relu') for _ in range(number_of_hidden)],  # Loop inside the list
    *[layer for _ in range(number_of_hidden-1) 
      for layer in (Dense(neurons, activation='relu'), Dropout(dropout_rate))
      ],
    Dense(1)
])

with tf.device('/GPU:0'):  # Force LSTM to run on GPU
    nn_model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
    nn_model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
    nn_train_preds = nn_model.predict(X_train)
    nn_rolling_preds = []
    nn_current_sequence = X_test[0].reshape(1, seq_length, 1)
    for _ in range(len(X_test)):
        next_pred = nn_model.predict(nn_current_sequence, verbose=0)[0, 0]
        nn_rolling_preds.append(next_pred)
        nn_current_sequence = np.roll(nn_current_sequence, -1, axis=1)
        nn_current_sequence[0, -1, 0] = next_pred

nn_test_preds = scaler.inverse_transform(np.array(nn_rolling_preds).reshape(-1, 1))
nn_train_preds = scaler.inverse_transform(nn_train_preds)

# inverse transform targets for plots later
y_train = scaler.inverse_transform(y_train.reshape(-1, 1))
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

#------------------------------------------ plots ------------------------------------------
# Define zoom-in indices
i = -200  # Start index for zooming (e.g., last 200 points)
j = None  # End index for zooming (use None for the end of the data)

# Get the starting and ending dates
start_date = data.index[i]  # Start date based on zoom-in index
end_date = data.index[j - 1] if j is not None else data.index[-1]  # End date (last date in the data)
end_date_with_buffer = end_date + pd.Timedelta(days=2)  # Add 2 days as a buffer

# Create subplots
fig, axes = plt.subplots(4, 1, figsize=(12, 8))  # 4 rows, 1 column

# Plot LR results
axes[0].plot(data.index[i:j], np.concatenate((y_lr_train, y_lr_test))[i:j], label="Actual Prices", color="blue", alpha=0.5)
axes[0].plot(data.index[seq_length:train_size+seq_length][i+10:j], lr_train_preds[i+10:j], label="LR Predictions (Train)", color="black", linestyle="--")
axes[0].plot(data.index[train_size+seq_length:][i:j], lr_preds[i:j], label="LR Predictions (Test)", color="red", linestyle="--")
axes[0].legend()
axes[0].set_title(f"{ticker} Stock Price Prediction")
axes[0].set_ylabel("Stock Price")
axes[0].grid(True, axis='x')  # Keep vertical grid lines
axes[0].grid(True, axis='y')  # Keep vertical grid lines
axes[0].tick_params(axis='x', labelbottom=False)  # Remove x-axis tick labels
axes[0].set_xlim([start_date, end_date_with_buffer])  # Set x-axis limits

# Plot GRU results
axes[1].plot(data.index[i:j], np.concatenate((y_train, y_test))[i:j], label="Actual Prices", color="blue", alpha=0.5)
axes[1].plot(data.index[seq_length:train_size+seq_length][i+10:j], gru_train_preds[i+10:j], label="GRU Predictions (Train)", color="black", linestyle="--")
axes[1].plot(data.index[train_size+seq_length:][i:j], gru_test_preds[i:j], label="GRU Predictions (Test)", color="red", linestyle="--")
axes[1].legend()
axes[1].set_ylabel("Stock Price")
axes[1].grid(True, axis='x')  # Keep vertical grid lines
axes[1].grid(True, axis='y')  # Keep vertical grid lines
axes[1].tick_params(axis='x', labelbottom=False)  # Remove x-axis tick labels
axes[1].set_xlim([start_date, end_date_with_buffer])  # Set x-axis limits

# Plot Simple Sequential NN results
axes[2].plot(data.index[i:j], np.concatenate((y_train, y_test))[i:j], label="Actual Prices", color="blue", alpha=0.5)
axes[2].plot(data.index[seq_length:train_size+seq_length][i+10:j], nn_train_preds[i+10:j], label="NN Predictions (Train)", color="black", linestyle="--")
axes[2].plot(data.index[train_size+seq_length:][i:j], nn_test_preds[i:j], label="NN Predictions (Test)", color="red", linestyle="--")
axes[2].legend()
axes[2].set_ylabel("Stock Price")
axes[2].grid(True, axis='x')  # Keep vertical grid lines
axes[2].grid(True, axis='y')  # Keep vertical grid lines
axes[2].tick_params(axis='x', labelbottom=False)  # Remove x-axis tick labels
axes[2].set_xlim([start_date, end_date_with_buffer])  # Set x-axis limits

# Plot LSTM results
axes[3].plot(data.index[i:j], np.concatenate((y_train, y_test))[i:j], label="Actual Prices", color="blue", alpha=0.5)
axes[3].plot(data.index[seq_length:train_size+seq_length][i+10:j], lstm_train_preds[i+10:j], label="LSTM Predictions (Train)", color="black", linestyle="--")
axes[3].plot(data.index[train_size+seq_length:][i:j], lstm_test_preds[i:j], label="LSTM Predictions (Test)", color="red", linestyle="--")
axes[3].legend()
axes[3].set_ylabel("Stock Price")
axes[3].grid(True, axis='x')  # Keep vertical grid lines
axes[3].grid(True, axis='y')  # Keep vertical grid lines
axes[3].tick_params(axis='x', labelbottom=True)  # Remove x-axis tick labels
axes[3].set_xlabel("Date")  # Keep x-axis label only for the last plot
axes[3].set_xlim([start_date, end_date_with_buffer])  # Set x-axis limits

# Adjust layout
plt.tight_layout()

# Save the figure as a PNG file in the "plots" directory
import os
plt.savefig(out_dir+"/stock_price_predictions.png", dpi=300)  # Save the figure with high resolution

plt.show()

# Evaluate and print MSE for all models
lr_train_rmse = (mean_squared_error(y_train, lr_train_preds))
lr_test_rmse = (mean_squared_error(y_test, lr_preds))
print(f"LR MSE (Train): {lr_train_rmse:.2f}")
print(f"LR MSE (Test): {lr_test_rmse:.2f}")

lstm_train_rmse = (mean_squared_error(y_train, lstm_train_preds))
lstm_test_rmse = (mean_squared_error(y_test, lstm_test_preds))
print(f"LSTM MSE (Train): {lstm_train_rmse:.2f}")
print(f"LSTM MSE (Test): {lstm_test_rmse:.2f}")

gru_train_rmse = (mean_squared_error(y_train, gru_train_preds))
gru_test_rmse = (mean_squared_error(y_test, gru_test_preds))
print(f"GRU MSE (Train): {gru_train_rmse:.2f}")
print(f"GRU MSE (Test): {gru_test_rmse:.2f}")

nn_train_rmse = (mean_squared_error(y_train, nn_train_preds))
nn_test_rmse = (mean_squared_error(y_test, nn_test_preds))
print(f"NN MSE (Train): {nn_train_rmse:.2f}")
print(f"NN MSE (Test): {nn_test_rmse:.2f}")

# Calculate residuals for each model
lr_train_residuals = y_train.flatten() - lr_train_preds.values
lr_test_residuals = y_test.flatten() - lr_preds.values

lstm_train_residuals = y_train - lstm_train_preds
lstm_test_residuals = y_test - lstm_test_preds

gru_train_residuals = y_train - gru_train_preds
gru_test_residuals = y_test - gru_test_preds

nn_train_residuals = y_train - nn_train_preds
nn_test_residuals = y_test - nn_test_preds

# Create a 2x2 grid for residual plots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Linear Regression Residuals
axes[0, 0].scatter(range(len(lr_train_residuals)), lr_train_residuals, label="Train Residuals", alpha=0.5, color="blue")
axes[0, 0].axhline(0, color="black", linestyle="--", linewidth=1)
axes[0, 0].set_title("Linear Regression Residuals")
axes[0, 0].set_xlabel("Index")
axes[0, 0].set_ylabel("Residuals")
axes[0, 0].legend()
axes[0, 0].grid(True)

# LSTM Residuals
axes[0, 1].scatter(range(len(lstm_train_residuals)), lstm_train_residuals, label="Train Residuals", alpha=0.5, color="blue")
axes[0, 1].axhline(0, color="black", linestyle="--", linewidth=1)
axes[0, 1].set_title("LSTM Residuals")
axes[0, 1].set_xlabel("Index")
axes[0, 1].set_ylabel("Residuals")
axes[0, 1].legend()
axes[0, 1].grid(True)

# GRU Residuals
axes[1, 0].scatter(range(len(gru_train_residuals)), gru_train_residuals, label="Train Residuals", alpha=0.5, color="blue")
axes[1, 0].axhline(0, color="black", linestyle="--", linewidth=1)
axes[1, 0].set_title("GRU Residuals")
axes[1, 0].set_xlabel("Index")
axes[1, 0].set_ylabel("Residuals")
axes[1, 0].legend()
axes[1, 0].grid(True)

# Neural Network Residuals
axes[1, 1].scatter(range(len(nn_train_residuals)), nn_train_residuals, label="Train Residuals", alpha=0.5, color="blue")
axes[1, 1].axhline(0, color="black", linestyle="--", linewidth=1)
axes[1, 1].set_title("Neural Network Residuals")
axes[1, 1].set_xlabel("Index")
axes[1, 1].set_ylabel("Residuals")
axes[1, 1].legend()
axes[1, 1].grid(True)

# Adjust layout and save the figure
plt.tight_layout()
plt.savefig(out_dir+"/train_residual_plots.png", dpi=300)  # Save the figure with high resolution
plt.show()

# Create a 2x2 grid for residual plots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Linear Regression Residuals
axes[0, 0].scatter(range(len(lr_test_residuals)), lr_test_residuals, label="Test Residuals", alpha=0.5, color="red")
axes[0, 0].axhline(0, color="black", linestyle="--", linewidth=1)
axes[0, 0].set_title("Linear Regression Residuals")
axes[0, 0].set_xlabel("Index")
axes[0, 0].set_ylabel("Residuals")
axes[0, 0].legend()
axes[0, 0].grid(True)

# LSTM Residuals
axes[0, 1].scatter(range(len(lstm_test_residuals)), lstm_test_residuals, label="Test Residuals", alpha=0.5, color="red")
axes[0, 1].axhline(0, color="black", linestyle="--", linewidth=1)
axes[0, 1].set_title("LSTM Residuals")
axes[0, 1].set_xlabel("Index")
axes[0, 1].set_ylabel("Residuals")
axes[0, 1].legend()
axes[0, 1].grid(True)

# GRU Residuals
axes[1, 0].scatter(range(len(gru_test_residuals)), gru_test_residuals, label="Test Residuals", alpha=0.5, color="red")
axes[1, 0].axhline(0, color="black", linestyle="--", linewidth=1)
axes[1, 0].set_title("GRU Residuals")
axes[1, 0].set_xlabel("Index")
axes[1, 0].set_ylabel("Residuals")
axes[1, 0].legend()
axes[1, 0].grid(True)

# Neural Network Residuals
axes[1, 1].scatter(range(len(nn_test_residuals)), nn_test_residuals, label="Test Residuals", alpha=0.5, color="red")
axes[1, 1].axhline(0, color="black", linestyle="--", linewidth=1)
axes[1, 1].set_title("Neural Network Residuals")
axes[1, 1].set_xlabel("Index")
axes[1, 1].set_ylabel("Residuals")
axes[1, 1].legend()
axes[1, 1].grid(True)

# Adjust layout and save the figure
plt.tight_layout()
plt.savefig(out_dir+"/test_residual_plots.png", dpi=300)  # Save the figure with high resolution
plt.show()

# Compute additional error metrics
def compute_error_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    return mse, rmse, mae

# Compute residual statistics
def compute_residual_stats(residuals):
    return {
        "Mean": np.mean(residuals),
        "Std Dev": np.std(residuals),
        "Min": np.min(residuals),
        "Max": np.max(residuals)
    }

# Calculate metrics for all models
error_metrics = {
    "Linear Regression": compute_error_metrics(y_test, lr_preds),
    "LSTM": compute_error_metrics(y_test, lstm_test_preds),
    "GRU": compute_error_metrics(y_test, gru_test_preds),
    "Neural Network": compute_error_metrics(y_test, nn_test_preds),
}

residual_stats = {
    "Linear Regression": compute_residual_stats(lr_test_residuals),
    "LSTM": compute_residual_stats(lstm_test_residuals),
    "GRU": compute_residual_stats(gru_test_residuals),
    "Neural Network": compute_residual_stats(nn_test_residuals),
}

# Save results to a text file
diagnostics_path = os.path.join(out_dir, "error_diagnostics.txt")
with open(diagnostics_path, "w") as f:
    f.write(f"Stock Prediction Error Diagnostics for {ticker}\n")
    f.write("="*50 + "\n\n")

    for model in error_metrics.keys():
        mse, rmse, mae = error_metrics[model]
        f.write(f"Model: {model}\n")
        f.write(f"  - Mean Squared Error (MSE): {mse:.4f}\n")
        f.write(f"  - Root Mean Squared Error (RMSE): {rmse:.4f}\n")
        f.write(f"  - Mean Absolute Error (MAE): {mae:.4f}\n")
        f.write(f"  - Residual Statistics:\n")
        for stat, value in residual_stats[model].items():
            f.write(f"    * {stat}: {value:.4f}\n")
        f.write("\n" + "-"*50 + "\n\n")

print(f"Error diagnostics saved to: {diagnostics_path}")