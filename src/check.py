import sqlite3
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()
DB_PATH = os.getenv('DB_PATH', '../data/stock_data.db')

conn = sqlite3.connect(DB_PATH)
df = pd.read_sql_query("SELECT * FROM model_features LIMIT 1", conn)
print("YOUR ACTUAL COLUMNS ARE:")
print(list(df.columns))
conn.close()