# Housing Price Prediction Model

This project implements a machine learning model for predicting housing prices using **regularized linear regression**. The model is designed to handle housing data with multiple features and employs ridge regression to prevent overfitting.

## Overview

The goal of this project is to predict housing prices based on various input features such as price, lot size, number of bedrooms, bathrooms, and stories. Regularized linear regression, specifically ridge regression, is employed to build a robust predictive model.

## Key Components

### Helper Functions

- `computeCost(X, y, theta, lam)`: Calculates the cost function for ridge regression.
- `denormalise_price(price)`: Denormalizes the predicted prices.
- `computeError(predicted, actual)`: Computes the error between predicted and actual prices.
- `plotGraph(x, y, labelX='X', labelY='Y', title='X vs Y')`: A utility function for plotting graphs.
- `gradientDescent(X, y, theta, iters, alpha, lam)`: Performs gradient descent for parameter optimization.
- `normalEquation(X, Y, lam)`: Computes the parameter vector using the normal equation.

### Model Training

- The model is trained using gradient descent with regularization.
- The optimal regularization parameter (lambda) is selected through a search over a range of lambda values.
- The model's performance is evaluated using the test dataset, and the mean absolute percentage error is computed.

### Results

- The optimal lambda value that minimizes the prediction error is reported.
- A graph depicting the relationship between lambda values and prediction error is included.
- An example prediction for a specific test case is shown.

### Lambda vs error using *gradient descent*
![Lambda vs error using gradient descent](https://github.com/iSHAANphaye/Housing-price-prediction/assets/75660041/c4ca684e-c318-4b82-af5b-41b1fafcddfb)

### Comparison with Normal Equation

- The model is also trained using the normal equation with regularization.
- The optimal lambda value is selected, and the model's performance is evaluated.
- A comparison between gradient descent and the normal equation is presented.

### Lambda vs error using *normal equation*
![Lambda vs error using normal equation](https://github.com/iSHAANphaye/Housing-price-prediction/assets/75660041/7503326f-cd5e-441f-b9e1-67b7c26513f9)

## Usage

1. Clone the repository to your local machine.
2. Install the necessary Python libraries and dependencies.
3. Run the provided Jupyter Notebook or Python script to train and evaluate the model.
4. Explore the model's predictions and lambda selection process.

## Dependencies

- Python 3.x
- NumPy
- Pandas
- Matplotlib
- Scikit-Learn (for train-test splitting)

## Acknowledgments

- This project is inspired by the need to predict housing prices accurately and efficiently.
- Special thanks to the open-source data science community for providing valuable insights and resources.

## To-do
- [ ] Implement better ML models to more accurately predict prices.
- [ ] Implement a website design to showcase model capabilities (Using Flask).

---
