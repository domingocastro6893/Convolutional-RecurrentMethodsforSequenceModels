import tensorflow as tf






# Print the dataset

# Step 1: Create a dataset with values from 0 to 9
dataset = tf.data.Dataset.range(10)
# Step 2: Create windows of size 5 with a shift of 1
dataset = dataset.window(5, shift=1, drop_remainder=True)
# Step 3: Flatten the windows into batches
dataset = dataset.flat_map(lambda window: window.batch(5))
# Step 4: Split each window into features and labels
dataset = dataset.map(lambda window: (window[:-1], window[-1:]))
# Step 5: Shuffle and batch the dataset
dataset = dataset.shuffle(buffer_size=10)
dataset = dataset.batch(2).prefetch(1)
# Print the dataset
for x,y in dataset:
    print("x = ", x.numpy())
    print("y = ", y.numpy())
