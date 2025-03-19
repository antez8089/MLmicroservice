import pandas as pd
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import pickle
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


class SimpleLightGBMModel:
    def __init__(self, file_path_sessions, file_path_tracks):
        file_path_tracks = os.path.join(BASE_DIR, "dane", file_path_tracks)
        file_path_sessions = os.path.join(BASE_DIR, "dane", file_path_sessions)
        self.file_path_sessions = file_path_sessions
        self.data_tracks = pd.read_json(file_path_tracks, lines=True)
        self.session_with_popularity = self.calculate_popularity()
        self.track_play_counts = {}
        self.model = None
        self.features = []
        self.target_date = None

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

    def prepare_features(self, target_date):
        target_date = pd.to_datetime(target_date)
        data = self.session_with_popularity[
            self.session_with_popularity["timestamp"] < target_date
        ]

        data = (
            data.set_index("timestamp")
            .groupby("track_id")
            .resample("D")
            .sum()
            .reset_index(drop=True)
        )
        data["lag_1"] = data.groupby("track_id")["play_count"].shift(1)
        data["lag_7"] = data.groupby("track_id")["play_count"].shift(7)
        data = data.dropna()

        total_play_counts = data.groupby("track_id")["play_count"].sum()
        for track_id, play_count in total_play_counts.items():
            self.track_play_counts[track_id] = play_count

        self.features = ["lag_1", "lag_7"]
        return data

    def train(self, target_date):
        self.target_date = pd.to_datetime(target_date)
        feature_df = self.prepare_features(self.target_date)
        X = feature_df[self.features]
        y = feature_df["play_count"]

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        params = {
            "objective": "regression",
            "metric": "rmse",
            "learning_rate": 0.01,
            "num_leaves": 60,
            "feature_fraction": 0.8,
        }

        self.model = lgb.train(
            params,
            train_data,
            valid_sets=[train_data, val_data],
            num_boost_round=4000,
        )

    def predict(self, target_date):
        target_date = pd.to_datetime(target_date)
        feature_df = self.prepare_features(target_date)
        X = feature_df[self.features]
        feature_df["gain"] = self.model.predict(X)

        predictions = feature_df.groupby("track_id")["gain"].sum().reset_index()

        for track_id in predictions["track_id"]:
            if track_id in self.track_play_counts:
                previous_play_count = self.track_play_counts[track_id]
                current_predicted_count = predictions.loc[
                    predictions["track_id"] == track_id, "gain"
                ].values[0]
                predictions.loc[predictions["track_id"] == track_id, "gain"] = (
                    current_predicted_count - previous_play_count
                )

        top_x = predictions.nlargest(20, "gain")
        track_mapping = self.data_tracks.set_index("id")["name"]
        top_x["name"] = top_x["track_id"].map(track_mapping)
        return top_x["name"].tolist()

    def save(self, file_name):
         file_path = os.path.join(BASE_DIR, "modele/trained_models", file_name)
        with open(file_path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(file_path):
         file_path = os.path.join(BASE_DIR, "modele/trained_models", file_name)
        with open(file_path, "rb") as f:
            instance = pickle.load(f)
        return instance

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
    model = SimpleLightGBMModel("sessions.jsonl", "tracks.jsonl")
    model.train("2023-01-01")
    model.save("simple_lightgbm_model-2.pkl")
    predicted = model.predict("2023-01-08")
    print(model.calculate_accuracy("2023 -01-08", 80, predicted))
