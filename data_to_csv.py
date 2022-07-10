import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


INPUT_PATH = "breast-cancer-wisconsin.data"
OUTPUT_PATH = "breast-cancer-wisconsin.csv"


def read_data(path):
    data = pd.read_csv(path)
    return data


def add_headers(dataset, headers):
    dataset.columns = headers
    return dataset


def data_file_to_csv():
    headers = ["CodeNumber", "ClumpThickness", "UniformityCellSize", "UniformityCellShape", "MarginalAdhesion",
               "SingleEpithelialCellSize", "BareNuclei", "BlandChromatin", "NormalNucleoli", "Mitoses",
               "CancerType"]

    dataset = read_data(INPUT_PATH)
    dataset = add_headers(dataset, headers)
    dataset.to_csv(OUTPUT_PATH, index=False)

    print ("File saved ...!")


def main():
    data_file_to_csv()
    
if __name__ == "__main__":
    main()