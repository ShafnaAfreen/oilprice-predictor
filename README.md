# Oil Price Prediction Using LSTM

This repository contains a machine learning model that predicts the future prices of crude oil using historical price data. The model uses a Long Short-Term Memory (LSTM) neural network to predict future prices based on the past 10 days of data.

## Project Overview

This project aims to predict the price of crude oil using time-series data. It uses historical oil price data from Yahoo Finance and builds an LSTM model to forecast future prices. The model is trained on data from 2015 to 2025 and is able to predict the oil price for the next day.

## Dependencies

This project requires the following Python packages:

- `numpy`
- `pandas`
- `yfinance`
- `matplotlib`
- `scikit-learn`
- `tensorflow`

### Setting up the Environment

1. **Clone this repository**:

    ```bash
    git clone https://github.com/YourUsername/oilprice-predictor.git
    cd oilprice-predictor
    ```

2. **Create a virtual environment** (optional but recommended):

    ```bash
    python3 -m venv oilprice-env
    source oilprice-env/bin/activate
    ```

3. **Install the required dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

    If you don't have a `requirements.txt` file, you can install them manually with:

    ```bash
    pip install numpy pandas yfinance matplotlib scikit-learn tensorflow
    ```

## Usage

### Training the Model

To train the model, run the following command:

```bash
python3 train.py
