# Stock Market Deep Learning Model

## Code Files

| File | Description |
|------|-------------|
| `app.py` | Streamlit web application |
| `predict_live.py` | Live prediction engine |
| `config.py` | API keys and config |
| `v4_lstm.py` | Dual-stream LSTM-GRU model |
| `price_only_basline.py` | Baseline model |
| `nlp_pipeline.py` | FinBERT sentiment pipeline |
| `v4_build_sequences.py` | Builds 3D tensors for training |
| `feature_engineering.py` | Technical indicators |
| `preprocess_features.py` | Data preprocessing |
| `fetch_news.py` | News API ingestion |
| `setup_db.py` | Database setup |
| `export_for_colab.py` | Export data to Google Colab |
| `import_finbert.py` | Import FinBERT scores |
| `import_from_collab.py` | Import files from Colab |

## Data Files

The following files are too large to include in this repository and must be generated locally by running the pipeline. They are saved in a Google Drive folder

| File | Size |
|------|------|
| `stock_data.db` | 160 MB |
| `finbert_scored_news.csv` | 10.8 MB |
| `v4_X_price.npy` | 401 MB |
| `v4_X_sentiment.npy` | 200 MB |
| `v4_y_target.npy` | 1.1 MB |
| `v4_meta_data.npy` | 11 MB |

## Setup

```bash
pip install -r requirements.txt
```

`.env` file with  API keys:
```
ALPACA_API_KEY=
ALPACA_SECRET_KEY=
NEWS_API_KEY=
```

## Run

```bash
streamlit run app.py
```
