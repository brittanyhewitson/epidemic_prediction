import os
import re
import sys
import dill
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from operator import itemgetter
from imblearn.over_sampling import SMOTENC
from sklearn.model_selection import train_test_split

from src.visualize_data import (
    view_data_balance,
)
from src.helpers import (
    cast_to_bool,
)
from src.templates import (
    RANDOM_SEED,
    DATA_CHOICES,
    REGEX,
)


def get_x_y(all_data, drop_categories=True):
    """
    Get the X and y arrays from the dataframe to be used by the models
    """
    if drop_categories:
        all_data = all_data.drop(["location", "date"], axis=1)

    X = all_data.drop(["zika_cases"], axis=1).values
    y = all_data["zika_cases"].values

    return X, y


def get_data(data_choice="small_data"):
    """
    Read in the appropriate file according to the data choice and produce the X and y
    arrays to be used by the models
    """
    # Set up the filepath
    if os.getcwd().endswith("src"):
        filepath = "../csv_data"
    else:
        filepath = "csv_data"

    # Check that the data choice is valid
    if data_choice not in DATA_CHOICES:
        raise Exception(f"Unrecognized data choice, please make sure data choice is from {DATA_CHOICES}")
    
    # Read dataset
    if data_choice == "small_data":
        full_path_X = os.path.join(filepath, "X.pkl")
        full_path_y = os.path.join(filepath, "y.pkl")
        # Read the small dataset
        with open(full_path_X, "rb") as f:
            X = dill.load(f)
        with open(full_path_y, "rb") as f:
            y = dill.load(f)
    elif data_choice == "by_date":
        by_date_data = []
        directory = os.path.join(filepath, "smote_by_date")
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
        # Sort according to date
        by_date_data = sorted(by_date_data, key=itemgetter('date'), reverse=False)
        return by_date_data
    else:
        if data_choice == "big_data":
            full_path = os.path.join(filepath, "07_feature_engineering_and_cleaning.csv")
        elif data_choice == "smote":
            full_path = os.path.join(filepath, "all_smote_data.csv")
        elif data_choice == "single_date":
            full_path = os.path.join(filepath, "smote_by_date", "2016-06-25.csv")
        
        # Read input data
        all_data = pd.read_csv(full_path)
        all_data["zika_cases"] = all_data["zika_cases"].apply(cast_to_bool)
        X, y = get_x_y(all_data)

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

    return data


def smote(data, save_data=False, filename=None, view_plots=True):
    """
    Use SMOTE to upsample the minority class
    """
    X, y = get_x_y(data, drop_categories=False)
    
    if view_plots:
        # Plot balance of input data
        view_data_balance(X, y, data_type="input data")

    # Check for severely imbalanced data
    min_num_samples = 0.05*len(y)
    if len(np.unique(y)) == 1:
        logging.warning("Dataset only has one class, skipping")
        return None
    elif len(y[y == 0]) < min_num_samples:
        logging.warning("Dataset is too imbalanced to oversample, skipping")
        return None
    elif len(y[y == 1]) < min_num_samples:
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
        smote_data.to_csv(f"../csv_data/{filename}.csv", index=False)
    
    return smote_data


def smote_by_date(all_data, save_data=False):
    """
    Group the overall data by date and perform SMOTE on each subset
    """
    grouped_by_date = all_data.groupby(by="date")

    # Split data into groups by date and store into a list
    for name, group in grouped_by_date:
        group = group.drop_duplicates()
        logging.info(f"Balancing {name}")
        smote_data = smote(
            group,
            save_data=save_data,
            filename=os.path.join("smote_by_date", name),
            view_plots=False
        )


def main():
    # Read in the data
    all_data = pd.read_csv("../csv_data/07_feature_engineering_and_cleaning.csv")
    all_data["zika_cases"] = all_data["zika_cases"].apply(cast_to_bool)      

    # Oversample using SMOTE by date
    smote_by_date(all_data, save_data=True)

    # Oversample using SMOTE on the entire dataset
    smote_data = smote(all_data, save_data=True, filename="all_smote_data")


if __name__=='__main__':
    main()