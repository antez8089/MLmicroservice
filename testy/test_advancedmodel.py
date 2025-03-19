import pytest
import os
import pandas as pd
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "modele"))
from modelZaawansowany import AdvancedModel


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TEST_DATA_DIR = os.path.join(BASE_DIR, "dane")


@pytest.fixture
def advanced_model():
    # Tworzy instancję AdvancedModel z plikami testowymi
    return AdvancedModel("sessions_test.jsonl", "tracks_test.jsonl")


def test_calculate_popularity(advanced_model):
    # Test obliczania popularności na podstawie danych testowych
    popularity_data = advanced_model.session_with_popularity

    assert not popularity_data.empty
    assert "play_count" in popularity_data.columns
    assert "track_id" in popularity_data.columns
    assert "timestamp" in popularity_data.columns


def test_find_top_popularity(advanced_model):
    # Test wyszukiwania najpopularniejszych utworów
    top_tracks = advanced_model.find_top_popularity(5, "2025-01-07")

    assert len(top_tracks) <= 5
    assert isinstance(top_tracks, list)


def test_save_and_load(advanced_model):
    file_path = "test_advanced_model.pkl"
    full_path = os.path.join(BASE_DIR, "modele/trained_models", file_path)

    try:
        advanced_model.train("2025-01-01")
        advanced_model.save(file_path)
        assert os.path.exists(full_path)

        loaded_model = AdvancedModel.load(file_path)
        assert loaded_model.train_date == advanced_model.train_date
        assert len(loaded_model.fitted_models) == len(advanced_model.fitted_models)
    finally:
        if os.path.exists(full_path):
            os.remove(full_path)  # Usuwanie pliku po teście


def test_predict_empty_data(advanced_model, mocker):
    # Test przewidywania przy braku danych
    mocker.patch.object(
        advanced_model,
        "session_with_popularity",
        new=pd.DataFrame(columns=["timestamp", "track_id", "play_count"]),
    )

    advanced_model.train("2025-01-01")
    predictions = advanced_model.predict("2025-01-08")

    assert predictions == []


def test_find_top_popularity_with_mocked_data(advanced_model, mocker):
    # Mockowanie popularności dla uproszczonego testu
    mock_popularity = pd.DataFrame(
        {
            "track_id": [1, 2, 3],
            "timestamp": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"]),
            "play_count": [100, 50, 25],
        }
    )
    mocker.patch.object(advanced_model, "session_with_popularity", new=mock_popularity)

    top_tracks = advanced_model.find_top_popularity(2, "2025-01-04")
    assert len(top_tracks) == 2
