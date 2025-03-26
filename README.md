# 🔧 LSTM Predictive Maintenance

A machine learning system that uses Long Short-Term Memory (LSTM) neural networks to predict equipment failures before they occur, enabling proactive maintenance scheduling.

## 📋 Description

This project implements a predictive maintenance solution using LSTM neural networks to analyze equipment operational data and predict potential failures. By identifying patterns that precede equipment breakdown, maintenance can be scheduled proactively, reducing downtime and maintenance costs.

The system processes historical equipment operation logs, extracts relevant features, and trains an LSTM model to predict the time remaining until failure (RUL - Remaining Useful Life).

## ✨ Features

- Data preprocessing and feature engineering for equipment operational logs
- Time series analysis using LSTM neural networks
- Prediction of equipment failure probability
- Calculation of Remaining Useful Life (RUL) for equipment
- Support for multiple machine IDs and failure classes

## 🔍 Prerequisites

- Python 3.6+
- pandas
- scikit-learn
- keras
- tensorflow
- CSV file with equipment operations logs (`equipment_operations_logs.csv`)

## 🚀 Setup Guide

1. Clone this repository:
   ```bash
   git clone https://github.com/corticalstack/LSTMPredictiveMaintenance.git
   cd LSTMPredictiveMaintenance
   ```

2. Install required dependencies:
   ```bash
   pip install pandas scikit-learn keras tensorflow
   ```

3. Place your equipment operations logs in the root directory as `equipment_operations_logs.csv`

## 📊 Data Format

The expected CSV file should contain the following columns:
- `id_source_primary_machine` (renamed to `id_machine`): Unique identifier for each machine
- `dt_ti_cycle_start` (renamed to `cycle_start`): Timestamp for cycle start
- `tx_delay_class_description` (renamed to `class`): Classification of operational state or failure mode

## 💻 Usage

Run the main script to process data and train the model:

```bash
python main.py
```

The script will:
1. Load and preprocess the equipment operations data
2. Perform feature engineering
3. Split data into training and testing sets
4. Train an LSTM model to predict equipment failures
5. Save the trained model

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
