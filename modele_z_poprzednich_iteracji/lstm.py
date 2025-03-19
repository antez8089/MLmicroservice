import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
     

class LSTMModel:
    def __init__(self, file_path_sessions, file_path_tracks):
        self.file_path_sessions = file_path_sessions
        self.data_tracks = pd.read_json(file_path_tracks, lines=True)
        self.train_date = None  # Zmienna do zapisywania ostatniej daty treningowej
        self.scaler = MinMaxScaler()
        self.model = Sequential() # Model na poziomie klasy

    def generate_time_series(self):
        track_popularity_ts = []
        for chunk in pd.read_json(
            self.file_path_sessions, lines=True, chunksize=100000
        ):
            play_events = chunk[chunk["event_type"] == "play"]
            play_events["timestamp"] = pd.to_datetime(play_events["timestamp"])
            grouped = (
                play_events.groupby(["track_id", pd.Grouper(key="timestamp", freq="D")])
                .size()
                .reset_index(name="play_count")
            )
            track_popularity_ts.append(grouped)

        track_popularity_ts = pd.concat(track_popularity_ts)
        track_popularity_ts = (
            track_popularity_ts.groupby(["track_id", "timestamp"]).sum().reset_index()
        )

        self.scaler.fit(track_popularity_ts[["play_count"]])
        track_popularity_ts["play_count"] = self.scaler.transform(
            track_popularity_ts[["play_count"]]
        )

        # Group the dataframe by 'track_id' and store each group's dataframe in a list
        self.dataframes_list = [
            group[["timestamp", "play_count"]].reset_index(drop=True)
            for _, group in track_popularity_ts.groupby("track_id")
        ]

        self.train_date = track_popularity_ts["timestamp"].max()
        return self.dataframes_list

    def train(self):
        self.model = Sequential()
        ts = self.generate_time_series()
        self.tr = []
        self.test = []
        for element in ts:
            self.tr.append(element[:-7])
            self.test.append(element[-7:])

        n_input = 70
        n_features = 1
        self.model.add(LSTM(100, activation="relu", input_shape=(n_input, n_features)))
        self.model.add(Dense(1))
        self.model.compile(optimizer="adam", loss="mse")

        print(len(self.tr))
        for i in range(len(self.tr)):
            # Przygotowanie generatora
            train_data = np.array(self.tr[i]["play_count"]).reshape(-1, 1)
            generator = TimeseriesGenerator(train_data, train_data, length=n_input, batch_size=1)
            self.model.fit(generator, epochs=10)

    def predict(self, target_date, amount=7):
        predict_date = pd.to_datetime(target_date)

        if self.train_date is None:
            raise ValueError("Model has not been trained yet.")

        predictions = {}

        for idx, element in enumerate(self.dataframes_list):
            # Przycięcie szeregu czasowego do target_date
            trimmed = element[element["timestamp"] <= predict_date]
            if len(trimmed) < 70:
                print(f"Track {idx} has insufficient data for prediction. Skipping.")
                continue

            # Pobranie ostatnich 150 wartości jako dane wejściowe
            last_known_data = np.array(trimmed["play_count"][-70:]).reshape((1, 70, 1))

            # Iteracyjne prognozowanie kolejnych dni
            forecast = []
            current_input = last_known_data
            for _ in range(amount):
                prediction = self.model.predict(current_input, verbose=0)
                forecast.append(prediction[0, 0])
                # Aktualizacja danych wejściowych
                current_input = np.append(current_input[:, 1:, :], [[[prediction[0, 0]]]], axis=1)



            # Przeskalowanie prognoz do oryginalnej skali
            forecast = self.scaler.inverse_transform(np.array(forecast).reshape(-1, 1))
            predictions[f"track_{idx}"] = forecast.flatten()

        # Obliczanie sumy prognoz dla każdego utworu
        total_forecasts = {track: np.sum(values) for track, values in predictions.items()}

        # Sortowanie utworów według sumy prognoz i wybór top 20
        top_tracks = sorted(total_forecasts.items(), key=lambda x: x[1], reverse=True)[:20]

        # Przygotowanie wyników w formie słownika z prognozami dla top 20 utworów
        top_predictions = {track: predictions[track] for track, _ in top_tracks}

        # Mapowanie track_id na nazwę utworu
        id_to_name = dict(zip(self.data_tracks["id"], self.data_tracks["name"]))

        # Tworzenie wynikowego słownika z nazwami utworów
        named_predictions = {
            id_to_name.get(track.split("_")[1], f"Unknown Track {track}")
            for track, values in top_predictions.items()
        }

        return named_predictions




if __name__ == "__main__":
    model = LSTMModel("sessions.jsonl", "tracks.jsonl")
    model.train()
    predictions = model.predict("2024-01-08", amount=7)
    print(predictions)
