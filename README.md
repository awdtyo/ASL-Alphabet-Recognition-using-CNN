# ASL Alphabet Recognition using Convolutional Neural Networks (CNN)

## Project Overview

This project implements a Convolutional Neural Network (CNN) for American Sign Language (ASL) alphabet recognition. The model is designed to classify images of hand gestures into one of 26 possible ASL alphabet letters (excluding J and Z, which involve motion).

The dataset consists of 28x28 grayscale images of various ASL hand gestures. The goal is to train a robust classification model that can accurately identify the corresponding alphabet letter from an input image.

## Dataset

The dataset used for this project is loaded from `train.csv` and `test.csv` files. Each row in these CSVs represents an image, with the first column being the `label` (0-25 for A-Z) and the subsequent 784 columns (`pixel1` to `pixel784`) representing the pixel values of the 28x28 grayscale image.

### Preprocessing Steps:
1.  **Loading Data**: `train.csv` and `test.csv` are loaded into pandas DataFrames.
2.  **Splitting Features and Labels**: The 'label' column is separated from the pixel data.
3.  **Reshaping Images**: Pixel data is reshaped from a 1D array of 784 values into a 2D array of 28x28, and then expanded to include a channel dimension (28x28x1) for CNN input.
4.  **Normalization**: Pixel values are scaled from the range [0, 255] to [0.0, 1.0] by dividing by 255.0.
5.  **One-Hot Encoding**: Labels are converted into a one-hot encoded format using `tensorflow.keras.utils.to_categorical` to match the `categorical_crossentropy` loss function requirement for multi-class classification.

## Model Architecture

The model is a Sequential CNN built using Keras, designed to extract features from the image data and classify them. The architecture consists of the following layers:

*   **Input Layer**: `Conv2D` with 32 filters, 3x3 kernel, ReLU activation, and `input_shape=(28, 28, 1)`.
*   **Batch Normalization**: Applied after each convolutional layer to stabilize and accelerate training.
*   **MaxPooling2D**: 2x2 pooling to reduce spatial dimensions and create translation invariance.
*   **Dropout**: Introduced after pooling layers to prevent overfitting.

This pattern of `Conv2D -> BatchNormalization -> MaxPooling2D -> Dropout` is repeated multiple times with increasing filter counts (32, 64, 128) to learn progressively more complex features.

*   **Flatten Layer**: Converts the 3D output of the convolutional layers into a 1D vector.
*   **Dense Layer (Hidden)**: A fully connected layer with 128 units and ReLU activation.
*   **Dropout**: Another dropout layer to prevent overfitting in the dense layers.
*   **Output Layer**: A final `Dense` layer with 26 units (one for each alphabet class) and `softmax` activation for multi-class probability distribution.

## Training Details

### Compilation:
*   **Optimizer**: `Adam` optimizer is used for its adaptive learning rate capabilities.
*   **Loss Function**: `categorical_crossentropy` is selected as the loss function, suitable for multi-class classification with one-hot encoded labels.
*   **Metrics**: `accuracy` is monitored during training to evaluate model performance.

### Training Process:
*   The model is trained for a maximum of 100 epochs.
*   **Early Stopping**: An `EarlyStopping` callback is used to monitor `val_loss` with a `patience` of 5 epochs. Training will stop if the validation loss does not improve for 5 consecutive epochs, and the best weights from the training run are restored.
*   **Validation**: `validation_data=(x_test, y_test)` is provided to monitor the model's performance on unseen data during training.

## Results (Training and Validation Loss)

The training history, including loss and validation loss, is visualized to observe the model's learning progress and identify potential overfitting. The plot helps in understanding how well the model generalizes to new data.
