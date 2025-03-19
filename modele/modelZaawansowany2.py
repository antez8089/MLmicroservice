import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from tqdm import tqdm
import pickle
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


class AdvancedModel:
    def __init__(self, file_path_sessions, file_path_tracks):
        file_path_tracks = os.path.join(BASE_DIR, "dane", file_path_tracks)
        file_path_sessions = os.path.join(BASE_DIR, "dane", file_path_sessions)
        self.file_path_sessions = file_path_sessions
        self.data_tracks = pd.read_json(file_path_tracks, lines=True)
        self.session_with_popularity = self.calculate_popularity()
        self.unique_trained_tracks = []
        self.train_date = 0
        self.fitted_models = {}
        self.train_session = 0

    def find_actual_top(self, number, target_date):
        target_date = pd.to_datetime(target_date)
        current_data = self.session_with_popularity[
            (self.session_with_popularity["timestamp"] < target_date)
            & (
                self.session_with_popularity["timestamp"]
                > target_date - pd.Timedelta(days=7)
            )
        ]
        current_data = (
            current_data.groupby("track_id")["play_count"]
            .sum()
            .reset_index()[["track_id", "play_count"]]
        )
        top_x = current_data.nlargest(number, "play_count")
        track_mapping = self.data_tracks.set_index("id")["name"]
        top_x["name"] = top_x["track_id"].map(track_mapping)
        return top_x["name"].tolist()

    def calculate_popularity(self):
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
        return track_popularity_ts

    def train(self, target_date):
        self.train_date = pd.to_datetime(target_date)
        track_popularity_ts = self.session_with_popularity[
            (self.session_with_popularity["timestamp"] <= self.train_date)
        ]
        self.train_session = (
            track_popularity_ts.groupby("track_id")["play_count"]
            .sum()
            .reset_index()[["track_id", "play_count"]]
        )
        unique_tracks_num = track_popularity_ts["track_id"].nunique()
        self.unique_trained_tracks = track_popularity_ts["track_id"].unique()
        with tqdm(total=unique_tracks_num, desc="Model training") as pbar:
            for track_id, group in track_popularity_ts.groupby("track_id"):
                ts = group.set_index("timestamp")["play_count"].asfreq("D").fillna(0)
                if len(ts) > 2:  # Upewniamy się, że mamy wystarczająco danych
                    model = ExponentialSmoothing(ts, trend="add", seasonal=None)
                    self.fitted_models[track_id] = model.fit()
                pbar.update(1)

    def predict(self, target_date):
        predictions = []
        predict_date = pd.to_datetime(target_date)
        difference = predict_date - self.train_date
        with tqdm(
            total=len(self.unique_trained_tracks), desc="Model predicting"
        ) as pbar:
            for track_id in self.unique_trained_tracks:
                forecast = (
                    self.fitted_models[track_id]
                    .forecast(steps=difference.days)[-7]
                    .sum()
                )
                predictions.append({"track_id": track_id, "gain": forecast})
                pbar.update(1)

        predictions_df = pd.DataFrame(predictions)
        top_x = predictions_df.nlargest(20, "gain")
        track_mapping = self.data_tracks.set_index("id")["name"]
        top_x["name"] = top_x["track_id"].map(track_mapping)
        return top_x["name"].tolist()

    def save(self, file_name):
        file_path = os.path.join(BASE_DIR, "modele/trained_models", file_name)
        with open(file_path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(file_path):
        file_path = os.path.join(BASE_DIR, "modele/trained_models", file_path)
        with open(file_path, "rb") as f:
            instance = pickle.load(f)
        return instance

    def calculate_accuracy(self, date, size, prediction):
        actual = self.find_actual_top(size, date)
        actual_set = set(actual)
        match_count = 0

        for string in prediction:
            if string in actual_set:
                match_count += 1
                actual_set.remove(string)
        return match_count / len(prediction)


if __name__ == "__main__":
    model = AdvancedModel.load("advanced_model.pkl")
    # model.train("2024-01-01")
    # model.save("advanced_model.pkl")
    print(model.train_date)
    # actual = model.find_top_popularity(100, "2024-01-08")
    # print(predicted)
    # print(actual)
