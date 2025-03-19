import sys
import os
import matplotlib.pyplot as plt
import pandas as pd
import random
import pickle
import seaborn as sns

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "modele"))
from modelZaawansowany2 import AdvancedModel
from modelBazowy import BasicModel

def run_tests():
    advanced_model = AdvancedModel.load("advanced_model.pkl")
    basic_model = BasicModel("sessions.jsonl", "tracks.jsonl")
    dates = []
    averagesa = []
    averagesb = []
    for i in range(10):
        start = pd.to_datetime("2024-1-1")
        end = pd.to_datetime("2024-10-30")

        delta_days = (end - start).days

        random_days = random.randint(0, delta_days)

        random_date = start + pd.Timedelta(days=random_days)

        new_date = random_date.strftime("%Y-%m-%d")
        dates.append(new_date)
        predicteda = advanced_model.predict(new_date)
        predictedb = basic_model.predict(new_date)
        averageb = basic_model.calculate_acuracy(new_date, predictedb)

        new_date = pd.to_datetime(new_date)
        new_date = new_date + pd.Timedelta(weeks=1)
        new_date = new_date.strftime("%Y-%m-%d")
        averagea = advanced_model.calculate_accuracy(new_date, 50, predicteda)
        
        averagesa.append(averagea)
        averagesb.append(averageb)

    sorted_data = sorted(zip(dates, averagesa, averagesb), key=lambda x: x[0])
    dates, averagesa, averagesb = zip(*sorted_data)
    
    return list(dates), list(averagesa), list(averagesb)

def plot_results(dates, averagesa, averagesb):
    sns.set_theme(style="whitegrid")
    
    data = pd.DataFrame({
        "Dates": dates,
        "Advanced Model Accuracy": averagesa,
        "Basic Model Accuracy": averagesb
    })

    data_melted = data.melt(id_vars="Dates", 
                            value_vars=["Advanced Model Accuracy", "Basic Model Accuracy"],
                            var_name="Model", 
                            value_name="Accuracy")


    plt.figure(figsize=(12, 6))
    sns.barplot(data=data_melted, x="Dates", y="Accuracy", hue="Model")
    

    plt.xticks(rotation=45, ha="right")
    

    plt.xlabel("Dates")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Comparison of Advanced and Basic Models")
    

    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(script_dir, "accuracy_comparison-10-2.png")
    
    # Save the plot to the script's directory
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()  # Close

if __name__ == "__main__":
    dates, averagesa, averagesb = run_tests()
    plot_results(dates, averagesa, averagesb)
