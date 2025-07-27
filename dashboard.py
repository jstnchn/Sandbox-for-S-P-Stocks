import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import plotly.graph_objs as go
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight

st.set_page_config(
    page_title="Interactive LSTM Sandbox for S&P 500 Stock Movement Modeling",
    layout="wide")

def load_data():
    sp500 = yf.Ticker("^GSPC").history(period="max")
    sp500 = sp500.loc["1990-01-01":].copy()
    sp500.drop(["Dividends", "Stock Splits"], axis=1, inplace=True)
    sp500["NextWeekClose"] = sp500["Close"].shift(-5)
    sp500["Target"] = (sp500["NextWeekClose"] > sp500["Close"]).astype(int)
    for lag in range(1, 8):
        sp500[f"Close_Lag_{lag}"] = sp500["Close"].shift(lag)
    sp500 = sp500.dropna()
    return sp500

def create_sequences(df, predictors, sequence_length=14):
    X, y, dates = [], [], []
    data = df[predictors].values
    target = df["Target"].values
    idx = df.index
    for i in range(sequence_length, len(df)):
        X.append(data[i-sequence_length:i])
        y.append(target[i])
        dates.append(idx[i])
    return np.array(X), np.array(y), dates

sp500 = load_data()
min_date = sp500.index.min().date()
max_date = sp500.index.max().date()
date_range = st.sidebar.date_input(
    "Date range for analysis",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)
if isinstance(date_range, tuple):
    start_date, end_date = date_range
else:
    start_date = date_range
    end_date = max_date
start_date = pd.to_datetime(start_date)
end_date = pd.to_datetime(end_date)
filtered_sp500 = sp500.loc[(sp500.index.date >= start_date.date()) & (sp500.index.date <= end_date.date())].copy()

st.sidebar.image("https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/72x72/1f4c8.png", width=80)
st.sidebar.header("Analysis Controls")
st.sidebar.subheader("LSTM Model Parameters")
sequence_length = st.sidebar.slider("Sequence Length (days)", 7, 30, 14, step=1)
lstm_units = st.sidebar.slider("LSTM Units", 8, 128, 32, step=8)
dropout_rate = st.sidebar.slider("Dropout Rate", 0.0, 0.5, 0.2, step=0.05)
epochs = st.sidebar.slider("Epochs", 1, 50, 10, step=1)
batch_size = st.sidebar.slider("Batch Size", 16, 256, 32, step=16)
threshold = st.sidebar.slider("Prediction Threshold for 'Up' (Class 1)", 0.1, 0.9, 0.5, step=0.01)
with st.sidebar.expander("About this Dashboard"):
    st.markdown("""
    This dashboard uses an LSTM neural network to predict **next week's** S&P 500 direction.
    - **Green dots**: correct predictions
    - **Red dots**: incorrect predictions
    - **Prediction threshold**: adjust the probability needed to predict an 'up' week
    - **Class balance**: Now using class_weight in Keras fit to address imbalance.
    """)

st.title("📈 Interactive LSTM Sandbox for S&P 500 Stock Movement Modeling")
st.caption("Powered by Streamlit, yfinance, and TensorFlow/Keras")

predictors = [f"Close_Lag_{lag}" for lag in range(1,8)]
X, y, dates = create_sequences(filtered_sp500, predictors, sequence_length=sequence_length)
split = int(len(X) * 0.7)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
dates_train, dates_test = dates[:split], dates[split:]

# Compute class weights for the training set
classes = np.unique(y_train)
class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
class_weight_dict = dict(zip(classes, class_weights))

# Model definition
model = Sequential([
    LSTM(lstm_units, input_shape=(sequence_length, len(predictors))),
    Dropout(dropout_rate),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Training with class_weight
with st.spinner("Training LSTM model..."):
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        validation_data=(X_test, y_test),
        class_weight=class_weight_dict
    )

# Prediction
proba = model.predict(X_test).flatten()
preds = (proba >= threshold).astype(int)

# Results
col1, col2 = st.columns([2, 1])
with col2:
    st.subheader("LSTM Results")
    st.write(f"Trained on {len(X_train)} sequences, tested on {len(X_test)} sequences.")
    st.code(classification_report(y_test, preds, digits=3), language="text")
    with st.expander("Show Confusion Matrix"):
        cm = confusion_matrix(y_test, preds)
        st.write(pd.DataFrame(cm, index=["Actual Down", "Actual Up"], columns=["Pred Down", "Pred Up"]))
    with st.expander("Training History"):
        st.line_chart(pd.DataFrame(history.history))

    st.markdown("**Training set class distribution:**")
    st.dataframe(pd.Series(y_train).value_counts().rename("count"))

with col1:
    st.subheader("S&P 500 Close Price & Prediction Accuracy")
    plot_df = pd.DataFrame({
        "Date": dates_test,
        "Close": filtered_sp500.loc[dates_test, "Close"].values,
        "Target": y_test,
        "Predictions": preds
    }).set_index("Date")
    plot_df["Correct"] = plot_df["Target"] == plot_df["Predictions"]
    colors = np.where(plot_df["Correct"], "green", "red")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=plot_df.index, y=plot_df["Close"],
        mode="lines", name="S&P 500 Close",
        line=dict(color="#00b4d8")
    ))
    fig.add_trace(go.Scatter(
        x=plot_df.index, y=plot_df["Close"],
        mode="markers", marker=dict(color=colors, size=8, opacity=0.4),
        name="Prediction Accuracy"
    ))
    fig.update_layout(
        height=400,
        legend=dict(orientation="h"),
        plot_bgcolor="#181825",
        paper_bgcolor="#181825",
        font=dict(color="#f8f8f2")
    )
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("Raw Data Table"):
        st.dataframe(plot_df.tail(30))

st.markdown("""
    <style>
    .stApp {background-color: #181825;}
    .stSidebar {background-color: #282a36;}
    h1, h2, h3, h4, h5, h6, .stMarkdown, .stCaption, .stText, .stSubheader, .stTitle {
        color: #f8f8f2 !important;
    }
    </style>
""", unsafe_allow_html=True)