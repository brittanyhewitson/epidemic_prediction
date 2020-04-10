import os
import re
import sys
import dill
import logging
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTENC
from collections import Counter
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from visualize_data import (
    view_data_balance,
)

'''
# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", 
    stream=sys.stderr, 
    level=logging.INFO
)
'''

# Set the random seed for splitting data
RANDOM_SEED = 123

REGEX = r"^\d+-\d+-\d+.csv$"

DATA_CHOICES = [
    "big_data",
    "small_data",
    "smote",
    "by_date",
]

def cast_to_bool(zika_cases):
    """
    """
    if zika_cases > 0:
        return 1
    else:
        return 0


def get_x_y(all_data, drop_categories=True):
    """
    """
    if drop_categories:
        all_data = all_data.drop(["location", "date"], axis=1)

    X = all_data.drop(["zika_cases"], axis=1).values
    y = all_data["zika_cases"].values

    return X, y


def get_data(data_choice="small_data"):
    """
    """
    # Check that the data choice is valid
    if data_choice not in DATA_CHOICES:
        raise Exception(f"Unrecognized data choice, please make sure data choice is from {DATA_CHOICES}")
    
    # Read dataset
    if data_choice == "big_data":
        # Read the big input data set
        all_data = pd.read_csv("csv_data/07_feature_engineering_and_cleaning.csv")
        all_data["zika_cases"] = all_data["zika_cases"].apply(cast_to_bool)
        X, y = get_x_y(all_data)
    elif data_choice == "smote":
        # Read data balanced with smote
        all_data = pd.read_csv("csv_data/all_smote_data.csv")
        all_data["zika_cases"] = all_data["zika_cases"].apply(cast_to_bool)
        X, y = get_x_y(all_data)
    elif data_choice == "small_data":
        # Read the small dataset
        with open("csv_data/X.pkl", "rb") as f:
            X = dill.load(f)
        with open("csv_data/y.pkl", "rb") as f:
            y = dill.load(f)
    elif data_choice == "by_date":
        by_date_data = []
        directory = os.path.join(os.getcwd(), "csv_data", "smote_by_date")
        for filename in os.listdir(directory):
            if re.match(REGEX, filename, re.I):
                full_path = os.path.join(directory, filename)
                data = pd.read_csv(full_path)
                X, y = get_x_y(data)

                # Split into training and testing
                X_train, X_test, y_train, y_test = train_test_split(
                        X, 
                        y,
                        random_state=RANDOM_SEED,
                        test_size=0.2
                    )
                # Store as a dict
                data = {
                    "date": filename.strip(".csv"),
                    "X": X,
                    "y": y,
                    "X_train": X_train,
                    "X_test": X_test,
                    "y_train": y_train,
                    "y_test": y_test,
                }

                # Add to list
                by_date_data.append(data)
        return by_date_data

    # Split into training and testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y,
        random_state=RANDOM_SEED,
        test_size=0.2
    )

    # Store as a dict
    data = {
        "date": "all_dates",
        "X": X,
        "y": y,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }

    return [data]


def smote(data, save_data=False, filename=None, view_plots=True):
    """
    """
    X, y = get_x_y(data, drop_categories=False)
    
    if view_plots:
        # Plot balance of input data
        view_data_balance(X, y, data_type="input data")

    # Check for severely imbalanced data
    if len(np.unique(y)) == 1:
        logging.warning("Dataset only has one class, skipping")
        return None
    elif len(y[y == 0]) < 6:
        logging.warning("Dataset is too imbalanced to oversample, skipping")
        return None
    elif len(y[y == 1]) < 6:
        logging.warning("Dataset is too imbalanced to oversample, skipping")
        return None

    categories = []
    for idx, col in enumerate(X[0]):
        if type(col) is str:
            categories.append(idx)

    # Oversample the minority using SMOTE
    oversample = SMOTENC(random_state=RANDOM_SEED, categorical_features=categories)
    X_upsampled, y_upsampled = oversample.fit_resample(X, y)

    if view_plots:
        # Plot results
        view_data_balance(X_upsampled, y_upsampled, data_type="SMOTE data")
    
    # Restructure as a dataframe
    columns = data.drop(["zika_cases"], axis=1).columns
    smote_data = pd.DataFrame(X_upsampled, columns=columns)
    smote_data["zika_cases"] = y_upsampled
    #print(len(all_smote_data))
    
    if save_data:
        smote_data.to_csv(f"csv_data/{filename}.csv", index=False)
    
    return smote_data


def smote_by_date(all_data, save_data=False):
    """
    """
    grouped_by_date = all_data.groupby(by="date")

    # Split data into groups by date and store into a list
    for name, group in grouped_by_date:
        group = group.drop_duplicates()
        logging.info(f"Balancing {name}")
        smote_data = smote(
            group,
            save_data=True,
            filename=os.path.join("smote_by_date", name),
            view_plots=False
        )


def main():
    # Read in the data
    all_data = pd.read_csv("csv_data/07_feature_engineering_and_cleaning.csv")
    all_data["zika_cases"] = all_data["zika_cases"].apply(cast_to_bool)      

    # Oversample using SMOTE by date
    smote_data_by_date = smote_by_date(all_data, save_data=True)

    # Oversample using SMOTE on the entire dataset
    smote_data = smote(all_data, save_data=True, filename="all_smote_data")


if __name__=='__main__':
    main()