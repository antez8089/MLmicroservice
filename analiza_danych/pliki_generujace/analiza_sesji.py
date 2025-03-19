import pandas as pd

# Wczytanie pliku JSONL do DataFrame
df = pd.read_json("sessions.jsonl", lines=True)

# Wyświetlenie pierwszych wierszy
df = df.drop(columns=["session_id", "user_id"])
print(df.head())
