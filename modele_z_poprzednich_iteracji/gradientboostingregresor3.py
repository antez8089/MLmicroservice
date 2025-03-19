import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import os
import pickle

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

class GlobalModel:
    def __init__(self, file_path_sessions, file_path_tracks):
        file_path_tracks = os.path.join(BASE_DIR, "dane", file_path_tracks)
        file_path_sessions = os.path.join(BASE_DIR, "dane", file_path_sessions)
        self.file_path_sessions = file_path_sessions
        self.data_tracks = pd.read_json(file_path_tracks, lines=True)
        self.model = GradientBoostingRegressor(learning_rate=0.2, n_estimators=100, max_depth=4)

    def prepare_data(self, target_date):
        target_date = pd.to_datetime(target_date)
        week_offsets = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]  # Define week offsets
        week_ranges = [(target_date - pd.Timedelta(weeks=offset), target_date - pd.Timedelta(weeks=offset - 1)) for offset in week_offsets]

        self.data_tracks = self.data_tracks[['id', 'name']]

        track_popularity = []
        for chunk in pd.read_json(
            self.file_path_sessions, lines=True, chunksize=100_000
        ):
            chunk["timestamp"] = pd.to_datetime(chunk["timestamp"])
            play_events = chunk[chunk["event_type"] == "play"]

            week_counts = []
            for start, end in week_ranges:
                week_data = play_events[(play_events["timestamp"] >= start) & (play_events["timestamp"] < end)]
                week_data["day"] = week_data["timestamp"].dt.date
                week_counts.append(
                    week_data.groupby(["track_id", "day"]).size().reset_index(name="play_count")
                )

            track_popularity.append(week_counts)

        # Combine data across chunks
        combined_week_data = [
            pd.concat([chunk[i] for chunk in track_popularity], ignore_index=True)
            for i in range(len(week_offsets))
        ]

        # Fill missing days with 0
        def fill_missing_days(data, start_date, end_date):
            all_days = pd.date_range(start=start_date, end=end_date).date

            data = data.groupby(["track_id", "day"], as_index=False)["play_count"].sum()

            filled_data = (
                data.set_index(["track_id", "day"])
                .reindex(
                    pd.MultiIndex.from_product(
                        [data["track_id"].unique(), all_days], names=["track_id", "day"]
                    ),
                    fill_value=0,
                )
                .reset_index()
            )
            return filled_data

        filled_week_data = [
            fill_missing_days(data, start, end - pd.Timedelta(days=1))
            for data, (start, end) in zip(combined_week_data, week_ranges)
        ]

        # Aggregate by track for model input
        weekly_summaries = [
            data.groupby("track_id")["play_count"].sum().reset_index()
            for data in filled_week_data
        ]

        # Merge all weeks' data
        data = weekly_summaries[0]
        for i, week_data in enumerate(weekly_summaries[1:], start=1):
            week_data = week_data.rename(columns={"play_count": f"play_count_week{i}"})
            data = data.merge(week_data, on="track_id", how="left")

        data = self.data_tracks.merge(data, left_on="id", right_on="track_id", how="left").fillna(0)
        print("\n")
        print("Columns in data:", data.columns)

        return data

    def train(self, target_date):
        target_date = pd.to_datetime(target_date)

        # Loop to train on multiple weeks
        start_date = target_date - pd.Timedelta(weeks=55)
        current_date = start_date

        while current_date < target_date:
            print(f"Training for target date: {current_date}")
            data = self.prepare_data(current_date)

            # X: Play counts from previous weeks, y: Play counts from the most recent week
            feature_columns = [col for col in data.columns if col.startswith("play_count_week")]
            feature_columns.remove("play_count_week1")
            X = data[feature_columns]
            y = data["play_count_week1"]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            self.model.fit(X_train, y_train)

            train_score = self.model.score(X_train, y_train)
            test_score = self.model.score(X_test, y_test)

            print(f"Train score: {train_score}, Test score: {test_score}")

            current_date += pd.Timedelta(weeks=1)

    def predict(self, target_date, top_n=20):
        if not self.model:
            raise ValueError("Model has not been trained yet.")

        data = self.prepare_data(target_date)
        week_columns = [col for col in data.columns if col.startswith("play_count_week")]
        for i in range(len(week_columns) - 1, 0, -1):
            data[week_columns[i]] = data[week_columns[i - 1]]

        # Drop the first week column (as it is shifted out of range)
        data = data.drop(columns=[week_columns[0]])
        feature_columns = [col for col in data.columns if col.startswith("play_count_week")]
        X = data[feature_columns]
        data["predicted_play_count"] = self.model.predict(X)

        top_tracks = data.nlargest(top_n, "predicted_play_count")
        track_mapping = self.data_tracks.set_index("id")["name"].to_dict()
        top_track_names = top_tracks["track_id"].map(track_mapping).tolist()

        print(f"Top {top_n} predicted tracks:")
        print(top_track_names)

        return top_track_names

    def save(self, file_path):
        file_path = os.path.join(BASE_DIR, "modele/trained_models", file_path)
        with open(file_path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(file_path):
        file_path = os.path.join(BASE_DIR, "modele/trained_models", file_path)
        with open(file_path, "rb") as f:
            instance = pickle.load(f)
        return instance


if __name__ == "__main__":
    model = GlobalModel("sessions.jsonl", "tracks.jsonl")
    model.train("2024-11-30")
    predictions = model.predict("2024-10-08")
    model.save("global_model-3.pkl")
    # model = GlobalModel.load("global_model-3.pkl")
    # predictions = model.predict("2024-12-21")
