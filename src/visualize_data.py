import os
import dill

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from operator import itemgetter
from sklearn.ensemble import ExtraTreesClassifier

from src.templates import (
    TEMP_FIELDS,
    PRECIP_FIELDS,
    DISTANCES_FIELDS,
)

from src.helpers import (
    cast_to_bool,

)

sns.set()


def create_hor_bar(df, save_fig=False, title=None):
    # Create the horizontal bar plot
    plt.barh(list(df.index), abs(df.values))
    plt.title(title)
    plt.tight_layout()
    # Save the figure if specified
    if save_fig:
        # If the output directory does not yet exist
        if not os.path.exists("output_images"):
            os.makedirs("output_images")
        title = title.replace(" ", "_")
        plt.savefig(f"output_images/{title}.png")
    # Show the bar plot and clear the figure afterwards
    plt.show()
    plt.clf()


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# INPUT DATA
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def plot_imbalance(zika_percent, non_zika_percent, data_type="input data", save_fig=False):
    """
    """
    # Create the plot
    plt.bar(["zika", "non-zika"], [zika_percent, non_zika_percent])
    plt.ylabel("Percent of dataset")
    plt.title(f"Balance of {data_type}")
    if save_fig:
        # If the output directory does not yet exist
        if not os.path.exists("output_images"):
            os.makedirs("output_images")
        data_type = data_type.replace(" ", "_")
        plt.savefig(f"output_images/data_balance_{data_type}.png")
    plt.show()


def view_data_balance(X, y, data_type, save_fig=False):
    """
    """
    # Find the two classes
    non_zika = y[y == 0]
    zika = y[y > 0]

    # Look at data to view imbalance
    non_zika_len = len(non_zika)
    zika_len = len(zika)
    all_len = len(y)
    non_zika_percent = (non_zika_len/all_len)*100
    zika_percent = (zika_len/all_len)*100

    # Print results to console
    print(f"Number of non-zika cases: {non_zika_len}")
    print(f"Number of zika cases: {zika_len}")
    print(f"Percent of non-zika cases: {non_zika_percent}")
    print(f"Percent of zika cases: {zika_percent}")
    print(f"Total number of rows: {all_len}")

    # Show plot
    plot_imbalance(zika_percent, non_zika_percent, data_type=data_type, save_fig=save_fig)


def plot_averages(df, save_fig=False):
    """
    Produces a bar plot of the average value for each column in the dataframe
    Some columns may have useless averages, such as latitude and longitude, etc.

    ARGS:
        df:         (dataframe) pandas dataframe containing the data to plot
        save_fig:   (bool) flag to indicate whether to save the plot as an image
    """
    # Drop columns that have a meaningless average
    temp_df = df[TEMP_FIELDS]
    precip_df = df[PRECIP_FIELDS]
    distances_df = df[DISTANCES_FIELDS]

    # Calculate the average for each column
    temp_avg = temp_df.mean(axis=0)
    precip_avg = precip_df.mean(axis=0)
    distances_avg = distances_df.mean(axis=0)
    
    create_hor_bar(temp_avg, save_fig=save_fig, title="temperature averages")
    create_hor_bar(precip_avg, save_fig=save_fig, title="precipitation averages")
    create_hor_bar(distances_avg, save_fig=save_fig, title="distances averages")
    

def plot_feature_output_correlation(all_data, save_fig=False):
    """
    """
    all_data["zika_cases"] = all_data["zika_cases"].apply(cast_to_bool)
    
    # Try simple correlation
    corrmat = all_data.corr()
    plt.subplots(figsize=(17,17))
    sns.heatmap(
        corrmat, 
        cmap=sns.color_palette("RdBu_r"), 
        square=True, 
        annot=True, 
        annot_kws={"size": 10}
    )
    plt.tight_layout()
    plt.show()
    plt.clf()
    
    # Try a tree
    X = all_data.drop(["zika_cases", "date", "location"], axis=1)
    y = all_data["zika_cases"]

    model = ExtraTreesClassifier(n_estimators=20)
    model.fit(X.values, y.values)
    feature_df = pd.DataFrame({
        "feature": list(X.columns),
        "importance": model.feature_importances_
    })
    feature_df = feature_df.sort_values(by="importance")
    plt.barh(feature_df["feature"], feature_df["importance"])
    plt.title("Feature Importance")
    plt.tight_layout()
    if save_fig:
        # If the output directory does not yet exist
        if not os.path.exists("output_images"):
            os.makedirs("output_images")
        plt.savefig("output_images/feature_correlation.png")
    plt.show()
    plt.clf()


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# RESULTS
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def plot_nn_heatmap(results, plot_type="test", save_fig=False):
    """
    """
    accuracies = list(map(itemgetter("accuracy"), results))
    params = list(map(itemgetter("test_params"), results))
    learning_rates = list(map(itemgetter("learning_rate"), params))
    optimizers = list(map(itemgetter("optimizer"), params))
    epochs = list(map(itemgetter("epochs"), params))
    batch_size = list(map(itemgetter("batch_size"), params))
    regularizer = list(map(itemgetter("regularization"), params))
    dropout = list(map(itemgetter("dropout"), params))

    if plot_type == "test":
        df = pd.DataFrame({
            "accuracy": accuracies,
            "epochs": epochs,
            "batch_size": batch_size
        })
        df = df.drop_duplicates()
        pivotted = df.pivot("batch_size", "epochs", "accuracy")
        filename = "{}_{}_{}".format(results[0]["name"], learning_rates[0], optimizers[0])
    elif plot_type == "regularization":
        df = pd.DataFrame({
            "accuracy": accuracies,
            "regularization": regularizer,
            "dropout": dropout
        })
        df = df.drop_duplicates(subset=["regularization", "dropout"])
        pivotted = df.pivot("dropout", "regularization", "accuracy")
        filename = "{}_{}_{}_{}_{}".format(results[0]["name"], epochs[0], batch_size[0], learning_rates[0], optimizers[0])
    elif plot_type == "optimizer":
        df = pd.DataFrame({
            "accuracy": accuracies,
            "learning_rate": learning_rates,
            "optimizer": optimizers
        })
        df = df.drop_duplicates(subset=["learning_rate", "optimizer"])
        pivotted = df.pivot("learning_rate", "optimizer", "accuracy")
        filename = "{}_{}_{}".format(results[0]["name"], epochs[0], batch_size[0])

    sns.heatmap(
        pivotted, 
        cmap=sns.color_palette("RdBu_r"), 
        square=True, 
        #annot=True, 
        #annot_kws={"size": 10}, 
        #vmin=0.6
        )
    title = results[0]["name"].replace("_", " ")
    plt.title(title)
    plt.tight_layout()
    if save_fig:
        # If the output directory does not yet exist
        if not os.path.exists("output_images"):
            os.makedirs("output_images")
        plt.savefig(f"output_images/{filename}_heatmap.png")
    #plt.show()
    plt.clf()


def plot_history(results, save_fig=False):
    """
    """
    for result in results:
        history = result["history"]
        plt.plot(range(1, (len(history.history['accuracy'])+1)), history.history['accuracy'])
        plt.plot(range(1, (len(history.history['val_accuracy'])+1)), history.history['val_accuracy'])
        #plt.ylim((0.5, 1.0))
        title = result["name"].replace("_", " ")
        plt.title('{} accuracy for epochs={} batch_size={}'.format(title, result["test_params"]["epochs"], result["test_params"]["batch_size"]))
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['train', 'test'], loc='upper right')
        plt.tight_layout()
        if save_fig:
            # If the output directory does not yet exist
            if not os.path.exists("output_images"):
                os.makedirs("output_images")
            filename = result["name"]
            plt.savefig(f"output_images/{filename}_history.png")
        #plt.show()
        plt.clf()


def plot_knn_bar(results, save_fig=False):
    """
    """
    params = list(map(itemgetter("params"), results))
    neighbours = list(map(itemgetter("n_neighbours"), params))
    accuracies = list(map(itemgetter("accuracy"), results))

    # Create the plot
    plt.bar(list(map(str, neighbours)), accuracies)
    plt.xlabel("Number of neighbours")
    plt.ylabel("Accuracy")
    plt.title("Accuracy by number of neighbours")
    if save_fig:
        # If the output directory does not yet exist
        if not os.path.exists("output_images"):
            os.makedirs("output_images")
        plt.savefig("output_images/knn_bar.png")
    #plt.show()
    plt.clf()
	

def plot_rf_bar(results, save_fig=False):
    """
    """
    params = list(map(itemgetter("params"), results))
    estimators = list(map(itemgetter("n_estimators"), params))
    accuracies = list(map(itemgetter("accuracy"), results))

    # Create the plot
    plt.bar(list(map(str, estimators)), accuracies)
    plt.xlabel("Number of estimators")
    plt.ylabel("Accuracy")
    plt.title("Accuracy by number of estimators")
    if save_fig:
        # If the output directory does not yet exist
        if not os.path.exists("output_images"):
            os.makedirs("output_images")
        plt.savefig("output_images/rf_bar.png")
    #plt.show()
    plt.clf()	
    

def plot_all_results(all_results, save_fig=False):
    """
    """
    best_results = []
    # Sort all results
    for result_list in all_results:
        # Sort according to accuracy
        sorted_list = sorted(result_list, key=itemgetter("accuracy"), reverse=True)
        best_results.append(sorted_list[0])

    accuracies = list(map(itemgetter("accuracy"), best_results))
    labels = list(map(itemgetter("name"), best_results))

    # Create the plot
    plt.bar(labels, accuracies)
    plt.xlabel("Classifier")
    plt.ylabel("Accuracy")
    plt.title("Accuracy by Classifier")
    if save_fig:
        # If the output directory does not yet exist
        if not os.path.exists("output_images"):
            os.makedirs("output_images")
        plt.savefig("output_images/all_classifiers.png")
    #plt.show()
    plt.clf()


def plot_accuracy_by_date(dates, results, save_fig=False):
    """
    """
    for key, val in results.items():
        plt.plot(dates, val, label=key)
    plt.legend(loc="lower right")
    plt.title("Classifier Accuracy by Date")
    plt.ylabel("Accuracy (%)")
    plt.xlabel("Date")
    plt.xticks(rotation=90)
    if save_fig:
        # If the output directory does not yet exist
        if not os.path.exists("output_images"):
            os.makedirs("output_images")
        plt.savefig("output_images/accuracy_by_date.png")
    #plt.show()
    plt.clf()


def plot_accuracy_by_date_subplot(dates, results, save_fig=False):
    """
    """
    num_plots = len(results)
    fig, axs = plt.subplots(num_plots, sharex=True, sharey=True, figsize=(15,10))
    i = 0
    for key, val in results.items():
        axs[i].plot(dates, val, label=key)
        axs[i].set_title(key.replace("_", " "))
        i += 1

    plt.xlabel("Date")
    plt.xticks(rotation=90)
    
    for ax in axs.flat:
        ax.label_outer()

    fig.text(0.002, 0.5, 'Accuracy (%)', va='center', rotation='vertical')
    plt.tight_layout()
    if save_fig:
        # If the output directory does not yet exist
        if not os.path.exists("output_images"):
            os.makedirs("output_images")
        plt.savefig("output_images/accuracy_by_date.png")
    #plt.show()
    plt.clf()


def main():
    # READ DATA
    #zika_cases = pd.read_csv("csv_data/03_infection_data_final.csv")
    all_data = pd.read_csv("../csv_data/07_feature_engineering_and_cleaning.csv")
    X = all_data.drop(["location", "date", "zika_cases"], axis=1).values
    y = all_data["zika_cases"].values
    
    # Read the small dataset
    with open("../csv_data/X.pkl", "rb") as f:
        X_small = dill.load(f)
    with open("../csv_data/y.pkl", "rb") as f:
        y_small = dill.load(f)

    # DATA VISUALUZATION
    # Plot data balance for large dataset
    view_data_balance(X, y, data_type="input data", save_fig=True)

    # Plot balance for small dataset
    view_data_balance(X_small, y_small, data_type="small dataset", save_fig=True)

    # Plot the averages for each data column
    plot_averages(all_data, save_fig=True)

    # Look at correlation between features and the number of cases
    plot_feature_output_correlation(all_data, save_fig=True)
    

if __name__=='__main__':
    main()