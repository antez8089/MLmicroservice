from flask import Flask, request, jsonify, redirect, url_for
import logging
import random
import sys
import os
import pickle
import hashlib
import pandas as pd
import json

# Dodanie folderu obok do sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "modele"))
from modelZaawansowany2 import AdvancedModel
from modelBazowy import BasicModel
from calculate_AB_result import calculate_count

app = Flask(__name__)  # Instancja aplikacji Flask
logging.getLogger("werkzeug").setLevel(logging.ERROR)

logging.basicConfig(
    filename="experiment.log",
    level=logging.INFO,
    force=True,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


class PostHandler:
    def __init__(self, app):
        self.app = app  # Przypisanie instancji aplikacji Flask
        self.add_routes()  # Dodanie endpointów
        self.generate_logs = False
        self.AdvancedModel = AdvancedModel.load("advanced_model.pkl")
        self.BasicModel = BasicModel("sessions.jsonl", "tracks.jsonl")
        self.predicted_modelA = []
        self.predicted_modelB = []
        self.file_path = "AB_users.json"
        self.AB_test_date = 0

        try:
            with open(self.file_path, "r") as file:
                json.load(file)
        except (FileNotFoundError, json.JSONDecodeError):
            self._initialize_file()
        self.ABtest_users = (
            self.read_AB_users()
        )  # Lista użytkowników biorących udział w teście AB

    def _initialize_file(self):
        data = {"groupA": [], "groupB": []}
        with open(self.file_path, "w") as file:
            json.dump(data, file, indent=4)

    def read_AB_users(self):
        with open(self.file_path, "r") as file:
            data = json.load(file)
        return [data["groupA"], data["groupB"]]

    def clear_AB_users(self):
        self.ABtest_users = [[], []]
        self._initialize_file()

    def add_AB_user(self, user, group):
        with open(self.file_path, "r") as file:
            data = json.load(file)
        if user not in data[group]:
            data[group].append(user)
        with open(self.file_path, "w") as file:
            json.dump(data, file, indent=4)

    def add_routes(self):
        self.app.add_url_rule(
            "/predict/modelA", "predictA", self.predictA, methods=["POST"]
        )
        self.app.add_url_rule(
            "/predict/modelB", "predictB", self.predictB, methods=["POST"]
        )
        self.app.add_url_rule(
            "/ABtest/begin", "ABtest_begin", self.begin_AB_test, methods=["POST"]
        )
        self.app.add_url_rule(
            "/ABtest/get_playlist",
            "ABtest_getplaylist",
            self.ABtest_getplaylist,
            methods=["POST"],
        )
        self.app.add_url_rule(
            "/ABtest/finish", "ABtest_finish", self.finish_AB_test, methods=["POST"]
        )

    def begin_AB_test(self):
        if len(self.ABtest_users[0]) + len(self.ABtest_users[1]) != 0:
            return "Najpierw zakończ poprzedni test AB. \n"

        try:
            data = request.get_json()
            self.AB_test_date = data["input_data"]
            if not data:
                return jsonify({"error": "Brak danych"}), 400

            logging.info(f'Test AB. Test rozpoczęty dla daty {data["input_data"]}.')

            predictionA = self.BasicModel.predict(data["input_data"])
            predictionB = self.AdvancedModel.predict(data["input_data"])

            self.predicted_modelA = predictionA
            self.predicted_modelB = predictionB

            logging.info(f"Test AB. Playlist model A: {predictionA}.")
            logging.info(f"Test AB. Playlist model B: {predictionB}.")
            return jsonify(
                {
                    "Data testu": data["input_data"],
                    "Playlist model A": predictionA,
                    "Playlist model B": predictionB,
                }
            )
        except Exception as e:
            logging.error(f"Błąd w predictA: {e}")
            return jsonify({"error": str(e)}), 500

    def finish_AB_test(self):
        averageA = calculate_count(
            self.ABtest_users[0],
            self.AB_test_date,
            self.predicted_modelA,
            "tracks.jsonl",
            "sessions.jsonl",
        ) / len(self.ABtest_users[0])

        averageB = calculate_count(
            self.ABtest_users[1],
            self.AB_test_date,
            self.predicted_modelB,
            "tracks.jsonl",
            "sessions.jsonl",
        ) / len(self.ABtest_users[1])

        if len(self.predicted_modelA) + len(self.predicted_modelB) == 0:
            return "Najpierw rozpocznij test AB.\n"
        logging.info("Test AB. Test zakończony.")
        logging.info(
            f"Test AB. Model A users: {self.ABtest_users[0]}. Users count: {len(self.ABtest_users[0])}."
        )
        logging.info(
            f"Test AB. Model B users: {self.ABtest_users[1]}. Users count: {len(self.ABtest_users[1])}."
        )

        self.clear_AB_users()
        return jsonify(
            {
                "Average play count A": averageA,
                "Average play count B": averageB,
            }
        )

    def predictA(self):
        self.generate_logs = True
        try:
            data = request.get_json()

            if not data:
                return jsonify({"error": "Brak danych"}), 400
            prediction = self.BasicModel.predict(data["input_data"])

            if self.generate_logs:
                logging.info(f"Model Bazowy, Input: {data}, Prediction: {prediction}")
                self.generate_logs = False
            return jsonify(
                {
                    "model_used": "Model Bazowy",
                    "prediction_time": data["input_data"],
                    "prediction": prediction,
                }
            )
        except Exception as e:
            logging.error(f"Błąd w predictA: {e}")
            return jsonify({"error": str(e)}), 500

    def predictB(self):
        self.generate_logs = True
        try:
            data = request.get_json()
            if not data:
                return jsonify({"error": "Brak danych"}), 400

            prediction = self.AdvancedModel.predict(data["input_data"])

            if self.generate_logs:
                logging.info(
                    f"Model Zaawansowany, Input: {data}, Prediction: {prediction}"
                )
                self.generate_logs = False
            return jsonify(
                {
                    "model_used": "Model Zaawansowany",
                    "prediction_time": data["input_data"],
                    "prediction": prediction,
                }
            )
        except Exception as e:
            logging.error(f"Błąd w predictB: {e}")
            return jsonify({"error": str(e)}), 500

    def ABtest_getplaylist(self):
        if len(self.predicted_modelA) + len(self.predicted_modelB) == 0:
            return self.predicted_modelA
        try:
            data = request.get_json()
            if not data:
                return jsonify({"error": "Brak danych"}), 400

            user = data["user_id"]
            group = 0
            print(self.ABtest_users)
            if user in self.ABtest_users[0]:
                group = "A"
            elif user in self.ABtest_users[1]:
                group = "B"
            else:
                group = self.hash_id(user)
                logging.info(f"Test AB. User {user} added to group {group}")
                if group == "A":
                    self.ABtest_users[0].append(user)
                    self.add_AB_user(user, "groupA")
                else:
                    self.ABtest_users[1].append(user)
                    self.add_AB_user(user, "groupB")
            if group == "A":
                return jsonify(
                    {
                        "Picked group": "A.Model Bazowy",
                        "Playlist": self.predicted_modelA,
                    }
                )
            else:
                return jsonify(
                    {
                        "Picked group": "B.Model Zaawansowany",
                        "Playlist": self.predicted_modelB,
                    }
                )
        except Exception as e:
            logging.error(f"Błąd w AB_test: {e}")
            return jsonify({"error": str(e)}), 500

    def hash_id(self, num_id):
        if int(hashlib.md5(str(num_id).encode()).hexdigest(), 16) % 2 == 0:
            return "A"
        else:
            return "B"


if __name__ == "__main__":
    handler = PostHandler(app)
    app.run(debug=True, host="0.0.0.0", port=8060)
