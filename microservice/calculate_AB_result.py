import pandas as pd
import os
from tqdm import tqdm

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def calculate_count(users_list, date, track_list, file_path_tracks, file_path_sessions):
    file_path_tracks = os.path.join(BASE_DIR, "dane", file_path_tracks)
    file_path_sessions = os.path.join(BASE_DIR, "dane", file_path_sessions)
    data_tracks = pd.read_json(file_path_tracks, lines=True)

    id_list = users_list
    count = 0
    track_id_list = []
    id_elem = 0
    for track in track_list:
        id_elem = data_tracks.loc[data_tracks["name"] == track, "id"]
        track_id_list.append(id_elem)

    for chunk in tqdm(
        pd.read_json(file_path_sessions, lines=True, chunksize=100000),
        desc="Przetwarzanie",
        unit="chunk",
    ):
        chunk["timestamp"] = pd.to_datetime(chunk["timestamp"])
        play_events = chunk[
            (chunk["event_type"] == "play")
            & (chunk["track_id"].isin(track_id_list))
            & (chunk["user_id"].isin(id_list))
            & (chunk["timestamp"] >= pd.to_datetime(date))
        ]
        count += play_events.shape[0]
    return count
