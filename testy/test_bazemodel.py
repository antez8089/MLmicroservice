import pytest
import os
import pandas as pd
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "modele"))

from modelBazowy import BasicModel

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TEST_DATA_DIR = os.path.join(BASE_DIR, "dane")


@pytest.fixture
def basic_model():
    # Przygotowuje instancję BasicModel z danymi testowymi
    return BasicModel("sessions_test.jsonl", "tracks_test.jsonl")


def test_init_model(basic_model):
    # Test, czy model został poprawnie zainicjalizowany
    assert isinstance(basic_model.data_tracks, pd.DataFrame)
    assert "name" in basic_model.data_tracks.columns
    assert os.path.exists(basic_model.file_path_sessions)


def test_predict(basic_model):
    # Test metody predict
    target_date = "2024-01-01"
    top_tracks = basic_model.predict(target_date, 5)
    assert len(top_tracks) == 5  # Sprawdza, czy zwraca odpowiednią liczbę wyników
    assert all(isinstance(track, str) for track in top_tracks)  # Sprawdza typ wyników


def test_calculate_accuracy(basic_model, mocker):
    # Mockowanie wyników `predict`
    mocker.patch.object(
        basic_model,
        "predict",
        side_effect=[
            ["Track A", "Track B", "Track C"],
            ["Track A", "Track C", "Track D", "Track E"],
        ],
    )

    predicted = ["Track A", "Track C", "Track X"]
    accuracy = basic_model.calculate_acuracy("2025-01-01", predicted)

    # Sprawdza poprawność obliczonej dokładności
    assert 0 <= accuracy <= 1
    assert accuracy == 2 / 3  # Oczekiwany wynik


def test_predict_empty_data(basic_model, mocker):
    # Mockowanie pustych danych wejściowych w postaci generatora
    empty_chunk = pd.DataFrame(columns=["timestamp", "event_type", "track_id"])
    mocker.patch(
        "pandas.read_json",
        return_value=(
            chunk for chunk in [empty_chunk]
        ),  # Generator zwracający DataFrame
    )

    top_tracks = basic_model.predict("2025-01-01", 5)

    assert top_tracks == []  # Brak wyników dla pustych danych
