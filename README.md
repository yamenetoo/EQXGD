# EQXGD
## train_ann_model Function

The `train_ann_model` function is designed to create, train, and evaluate an Artificial Neural Network (ANN) model using input data. This function utilizes TensorFlow and Keras libraries to facilitate the training of the model for predicting parameters (theta, alpha, beta) based on provided input features.

### Functionality
- **Data Splitting**: The function splits the provided dataset into training and test sets, ensuring a robust evaluation of the model.
- **Data Normalization**: It applies standard scaling to the input features to improve the model's performance.
- **Model Architecture**: The ANN model consists of three hidden layers with ReLU activation functions, culminating in an output layer designed to predict three parameters.
- **Training and Evaluation**: The model is trained using the Adam optimizer and mean squared error loss function. It also evaluates the model on the test set to compute the test loss.
- **Model Persistence**: The trained model can be saved to a specified path for future use.

### Parameters
- `X (numpy.ndarray)`: A 2D array of input features where each row corresponds to a sample.
- `Y (numpy.ndarray)`: A 2D array of target outputs where each row contains the parameters (theta, alpha, beta) associated with the samples.
- `model_save_path (str)`: The file path to save the trained model. Defaults to 'ann_model.h5'.
- `epochs (int)`: The number of epochs for training the model. Default is 2000.
- `batch_size (int)`: The number of samples processed before the model is updated. Default is 64.
- `test_size (float)`: The proportion of the dataset used for testing. Default is 0.2.
- `random_state (int)`: Seed for random number generation to ensure reproducibility. Default is 123.

### Returns
- `model`: The trained ANN model.
- `history`: The training history containing loss values and metrics over epochs.
- `predictions`: The predictions made by the model on the test set.
- `actual`: The actual values of the parameters (theta, alpha, beta) from the test set.

### Example Usage
```python
X, Y = generate_EQXGD_data(n_samples=10000, n_per_sample=30)
model, history, predictions, actual = train_ann_model(X, Y)
