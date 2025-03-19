# IUM Project

## Authors:
Antek Grajek  

## Project Overview:
This microservice is designed for predictive modeling in the music streaming domain. It allows users to make predictions using both a basic and an advanced model, as well as conduct A/B testing for playlist recommendations. The service processes data from three JSONL files (`artists.jsonl`, `sessions.jsonl`, and `tracks.jsonl`), which must be placed in the `data` directory before use.

## Data:
To ensure proper functionality of the models, the following data files must be provided in the `data` folder within the project directory:
- `artists.jsonl`
- `sessions.jsonl`
- `tracks.jsonl`

## Microservice Startup Instructions:
To activate the microservice, follow these steps:

### First-time setup:
```
cd microservice
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 post_handler.py
```

### Subsequent launches:
```
cd microservice
source .venv/bin/activate
python3 post_handler.py
```

## Request Handling Guide:
The microservice provides the following functionalities for example data:

### **Prediction using the basic model:**
```
curl -X POST -H "Content-Type: application/json" -d '{"input_data": "2024-01-24"}' --max-time 360 http://localhost:8060/predict/modelA
```

### **Prediction using the advanced model:**
```
curl -X POST -H "Content-Type: application/json" -d '{"input_data": "2024-01-24"}' --max-time 360 http://localhost:8060/predict/modelB
```

### **A/B Testing - Start tests:**
```
curl -X POST -H "Content-Type: application/json" -d '{"input_data": "2024-01-08"}' --max-time 360 http://localhost:8060/ABtest/begin
```

### **A/B Testing - Assign user to a group:**
```
curl -X POST -H "Content-Type: application/json" -d '{"user_id": "134"}' --max-time 360 http://localhost:8060/ABtest/get_playlist
```

### **A/B Testing - End tests:**
```
curl -X POST -H "Content-Type: application/json" --max-time 360 http://localhost:8060/ABtest/finish
```

## Test Report:
Additional information on the conducted tests is stored in the `experiments.log` file.

No pretrained model included on purpose of saving github memory.