import pandas as pd
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


class BasicModel:
    def __init__(self, file_path_sessions, file_path_tracks):
        file_path_tracks = os.path.join(BASE_DIR, "dane", file_path_tracks)
        file_path_sessions = os.path.join(BASE_DIR, "dane", file_path_sessions)
        self.data_tracks = pd.read_json(file_path_tracks, lines=True)
        self.file_path_sessions = file_path_sessions

    def predict(self, target_date, ammount=20):
        track_mapping = self.data_tracks.set_index("id")["name"].to_dict()

        track_popularity = pd.Series(dtype=int)

        chunksize = 100_000
        for chunk in pd.read_json(
            self.file_path_sessions, lines=True, chunksize=chunksize
        ):
            chunk["timestamp"] = pd.to_datetime(chunk["timestamp"])

            max_date = pd.to_datetime(target_date)
            one_week_ago = max_date - pd.Timedelta(weeks=1)
            last_week_data = chunk[chunk["timestamp"] >= one_week_ago]
            last_week_data = last_week_data[last_week_data["timestamp"] < max_date]
            play_events = last_week_data[last_week_data["event_type"] == "play"]

            track_popularity = track_popularity.add(
                play_events["track_id"].value_counts(), fill_value=0
            )

        top_20_tracks = track_popularity.nlargest(ammount).index.tolist()
        top_20_track_names = [
            track_mapping.get(track_id, "Unknown") for track_id in top_20_tracks
        ]

        return top_20_track_names

    def calculate_acuracy(self, target_date, predicted):
        target_date_dt = pd.to_datetime(target_date)

        new_target_date_dt = target_date_dt + pd.Timedelta(weeks=1)

        actual_time = new_target_date_dt.strftime("%Y-%m-%d")
        actual = self.predict(actual_time, 50)
        actual_set = set(actual)
        counter = 0
        for string in predicted:
            if string in actual_set:
                counter += 1
                actual_set.remove(string)
        return counter / len(predicted)


if __name__ == "__main__":
    model = BasicModel("sessions.jsonl", "tracks.jsonl")
    # model.predict()
