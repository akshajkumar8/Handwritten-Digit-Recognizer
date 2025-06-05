# Import required libraries
import tensorflow as tf          # Deep learning framework
from tensorflow import keras     # High-level neural network API
from tensorflow.keras import layers  # Neural network building blocks
import numpy as np              # Numerical operations and array handling

# Load MNIST dataset (60,000 training images, 10,000 test images of handwritten digits)
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalize pixel values from [0-255] to [0-1] for better training
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Add channel dimension for CNN input (shape: [samples, height, width, channels])
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

# Print dataset information
print(f'x_train shape: {x_train.shape}')
print(f'{x_train.shape[0]} train samples')
print(f'{x_test.shape[0]} test samples')

# Build CNN model with the following architecture:
model = keras.Sequential([
    # First convolutional block
    layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),  # Extract basic features
    layers.BatchNormalization(),  # Normalize activations for stable training
    layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),  # Extract more complex features
    layers.BatchNormalization(),  # Normalize activations
    layers.MaxPooling2D(pool_size=(2, 2)),  # Reduce spatial dimensions
    layers.Dropout(0.25),  # Prevent overfitting
    
    # Second convolutional block
    layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),  # Extract higher-level features
    layers.BatchNormalization(),
    layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),  # Final feature extraction
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),  # Further reduce dimensions
    layers.Dropout(0.25),  # Prevent overfitting
    
    # Fully connected layers for classification
    layers.Flatten(),  # Convert 2D feature maps to 1D vector
    layers.Dense(256, activation='relu'),  # Dense hidden layer with 256 neurons
    layers.BatchNormalization(),  # Normalize activations
    layers.Dropout(0.5),  # Strong dropout for dense layer
    layers.Dense(10, activation='softmax')  # Output layer (10 classes, one per digit)
])

# Configure model training parameters
model.compile(
    optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.001),  # Adam optimizer with 0.001 learning rate
    loss='sparse_categorical_crossentropy',  # Standard loss for multi-class classification
    metrics=['accuracy']  # Track accuracy during training
)

# Train the model on MNIST data
print('Training model...')
history = model.fit(
    x_train, y_train,
    batch_size=128,  # Process 128 images per training step
    epochs=12,  # Train for 12 complete passes through the dataset
    validation_split=0.1,  # Use 10% of training data for validation
    verbose=1  # Show training progress
)

# Evaluate model performance on test set
test_scores = model.evaluate(x_test, y_test, verbose=0)
print(f'Test loss: {test_scores[0]:.4f}')
print(f'Test accuracy: {test_scores[1]:.4f}')

# Save trained model to file for later use
model.save('mnist.h5')
print('Model saved as mnist.h5')
