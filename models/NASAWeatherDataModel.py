import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import kerastuner as kt

# Function to read and preprocess data
def get_data():
    # Path to your CSV file
    data_file = "C:/Users/14014/kenzie/CreatingMLModelsToPredictSequences/graph.csv"

    # Read the file content
    with open(data_file, 'r') as f:
        data = f.read()

    lines = data.split('\n')
    anomalies = []

    # Extract anomalies from CSV data
    for line in lines:
        if line:
            linedata = line.split(',')
            if len(linedata) > 1 and linedata[1].strip().lower() != 'anomaly':  # Check length and skip header row
                try:
                    anomaly = float(linedata[1])
                    anomalies.append(anomaly)
                except ValueError:
                    continue  # Skip non-numeric rows

    # Convert anomalies list to NumPy array
    series = np.asarray(anomalies)
    time = np.arange(len(anomalies), dtype="float32")

    # Normalize the data
    mean = series.mean(axis=0)
    series -= mean
    std = series.std(axis=0)
    series /= std

    return time, series

# Get the normalized data
time, series = get_data()

# Splitting into training and validation sets
split_time = int(len(series) * 0.8)
time_train = time[:split_time]
x_train = series[:split_time]
time_valid = time[split_time:]
x_valid = series[split_time:]

# Function to create windowed dataset
def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    series = tf.expand_dims(series, axis=-1)  # Ensure series has a single dimension
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
    dataset = dataset.shuffle(shuffle_buffer).map(lambda window: (window[:-1], window[-1]))
    dataset = dataset.batch(batch_size).prefetch(1)
    return dataset

# Define parameters for windowed datasets
window_size = 24
batch_size = 12
shuffle_buffer_size = 48

# Create windowed datasets
train_dataset = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)
valid_dataset = windowed_dataset(x_valid, window_size, batch_size, shuffle_buffer_size)

# Function to build the Conv1D model
def build_model(hp):
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Conv1D(
        filters=hp.Int('filters', min_value=32, max_value=128, step=32),
        kernel_size=hp.Int('kernel_size', min_value=3, max_value=9, step=2),
        strides=hp.Int('strides', min_value=1, max_value=3, step=1),
        padding='causal', activation='relu', input_shape=[None, 1]
    ))

    model.add(tf.keras.layers.Dense(10, activation='relu'))
    model.add(tf.keras.layers.Dense(1))

    model.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3))
    return model

# Initialize the tuner (RandomSearch in this case)
tuner = kt.RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=5,
    executions_per_trial=3,
    directory='my_dir',
    project_name='cnn_tune'
)

# Perform the hyperparameter search
tuner.search(train_dataset, epochs=10, validation_data=valid_dataset)

# Get the best hyperparameters found during the search
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

# Build the model with the best hyperparameters
model = tuner.hypermodel.build(best_hps)

# Train the model with the best hyperparameters
history = model.fit(train_dataset, epochs=50, validation_data=valid_dataset)

# Function to make forecasts using the trained model
def model_forecast(model, series, window_size):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size))
    ds = ds.batch(32).prefetch(1)
    forecast = model.predict(ds)
    return forecast

# Making predictions
forecast = model_forecast(model, series[..., np.newaxis], window_size)
results = forecast[split_time - window_size:-1, -1, 0]

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(time_valid, x_valid, label='Actual')
plt.plot(time_valid, results, label='Forecast')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.show()
