import os
import dill
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import ExtraTreesClassifier
from operator import itemgetter

sns.set()

ZIKA_DATAFIELD_TO_KEEP = [
    "zika_confirmed_laboratory",
    "zika_reported_local",
    "total_zika_confirmed_autochthonous",
    #"total_zika_confirmed_imported",
    "zika_lab_positive",
    "cumulative_confirmed_local_cases",
    #"weekly_zika_confirmed",
    "gbs_confirmed_cumulative",
]

TEMP_FIELDS = [
    "max_temp", 
    "max_temp1", 
    "max_temp2",
    "mean_temp",
    "mean_temp1",
    "mean_temp2",
    "min_temp",
    "min_temp1",
    "min_temp2",
    "dew_point",
    "dew_point1",
    "dew_point2"
]

PRECIP_FIELDS = [
    "precipitation",
    "precipitation1",
    "precipitation2"
]

DISTANCES_FIELDS = [
    "airport_dist_any",
    "airport_dist_large",
    "mosquito_dist"
]



# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def get_data_field(df):
    """
    """
    return df["data_field"] in ZIKA_DATAFIELD_TO_KEEP


def cast_to_bool(zika_cases):
    """
    """
    if zika_cases > 0:
        return 1
    else:
        return 0


def clean_zika_data(zika_cases):
    """
    """
    # Only keep the desired data fields
    zika_cases = zika_cases[zika_cases.apply(get_data_field, axis=1)].reset_index()
    return zika_cases


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


def view_data_balance(X, y, data_type):
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
    plot_imbalance(zika_percent, non_zika_percent, data_type=data_type, save_fig=True)

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
    plt.subplots(figsize=(20,15))
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
def plot_simple_classifier_scatter(results, save_fig=False):
    """
    Don't use in script anymore
    """
    # Scatter plot for all variations
    accuracies = []
    batches = []
    epochs = []
    n=0
    for result in results:
        accuracies.append(round(result["accuracy"]*10, 2))
        batches.append(result["params"]["batch_size"])
        epochs.append(result["params"]["epochs"])
    
    # Normalize the accuracies
    min_val = min(accuracies)
    max_val = max(accuracies)
    normalized = np.round((accuracies-min_val)/(max_val-min_val), 2)*100
    
    fig, ax = plt.subplots()
    scatter = ax.scatter(batches, epochs, s=normalized)
    handles, labels = scatter.legend_elements(prop="sizes", alpha=0.5)
    legend = ax.legend(handles, labels, loc="upper right", title="Sizes")
    plt.xlabel("Number of Batches")
    plt.ylabel("Number of Epochs")
    plt.title("Accuracy for Simple Classifier")
    if save_fig:
        # If the output directory does not yet exist
        if not os.path.exists("output_images"):
            os.makedirs("output_images")
        plt.savefig("output_images/simple_scatter.png")
    plt.show()
    plt.clf()


def plot_simple_classifier_heatmap(results, save_fig=False):
    """
    """

    accuracies = list(map(itemgetter("accuracy"), results))
    params = list(map(itemgetter("params"), results))
    epochs = list(map(itemgetter("epochs"), params))
    batches = list(map(itemgetter("batch_size"), params))

    df = pd.DataFrame({
        "accuracy": accuracies,
        "epochs": epochs,
        "batch_size": batches
    })

    df = df.drop_duplicates()
    pivotted = df.pivot("batch_size", "epochs", "accuracy")

    sns.heatmap(
        pivotted, 
        cmap=sns.color_palette("RdBu_r"), 
        square=True, 
        #annot=True, 
        #annot_kws={"size": 10}, 
        #vmin=0.6
        )
    plt.title("Simple Classifier")
    plt.tight_layout()
    if save_fig:
        # If the output directory does not yet exist
        if not os.path.exists("output_images"):
            os.makedirs("output_images")
        plt.savefig("output_images/simple_heatmap.png")
    plt.show()
    plt.clf()
    

def plot_simple_classifier_bar(results, save_fig=False):
    """
    Don't use in script anymore
    """
    # Bar plot for each parameter tested for simple classifier
    contraints = list(set(map(itemgetter("varied"), results)))
    
    # Make bar plots
    for constraint in contraints:
        accuracies = []
        labels = []
        for result in results:
            if result["varied"] == constraint:
                accuracies.append(round(result["accuracy"], 2))
                labels.append(result["params"][constraint])
        
        # Create the plot
        plt.bar(list(map(str, labels)), accuracies)
        plt.xlabel(constraint)
        plt.ylabel("Accuracy")
        plt.title("Accuracy by {}".format(constraint.strip("s").replace("_", " ")))
        if save_fig:
            # If the output directory does not yet exist
            if not os.path.exists("output_images"):
                os.makedirs("output_images")
            plt.savefig("output_images/simple_bar.png")
        plt.show()
        plt.clf()


def plot_history(results):
    """
    """
    for result in results:
        history = result["history"]
        #history.history['accuracy'][0] = 0.6
        #history.history['val_accuracy'][0] = 0.55
        plt.plot(range(1, (len(history.history['accuracy'])+1)), history.history['accuracy'])
        plt.plot(range(1, (len(history.history['val_accuracy'])+1)), history.history['val_accuracy'])
        #plt.ylim((0.5, 1.0))
        plt.title('Model Accuracy for epochs={} batch_size={}'.format(result["params"]["epochs"], result["params"]["batch_size"]))
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['train', 'test'], loc='upper right')
        plt.show()


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
    plt.show()
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
    plt.show()
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
    plt.show()
    plt.clf()


def main():
    # READ DATA
    #zika_cases = pd.read_csv("csv_data/03_infection_data_final.csv")
    all_data = pd.read_csv("csv_data/07_feature_engineering_and_cleaning.csv")
    X = all_data.drop(["location", "date", "zika_cases"], axis=1).values
    y = all_data["zika_cases"].values
    
    # Read the small dataset
    with open("csv_data/X.pkl", "rb") as f:
        X_small = dill.load(f)
    with open("csv_data/y.pkl", "rb") as f:
        y_small = dill.load(f)


    # DATA VISUALUZATION
    # Plot data balance for large dataset
    view_data_balance(X, y, data_type="input data")

    # Plot balance for small dataset
    view_data_balance(X_small, y_small, data_type="small dataset")

    # Plot the averages for each data column
    plot_averages(all_data)

    # Look at correlation between features and the number of cases
    plot_feature_output_correlation(all_data)

    # Geographically look at the cities included in the data set
    # TODO???

    # Weather visualization
    # TODO???

    # Geographical visualization of population density
    # TODO???

    # Geographical representation of mosquito sightings
    # TODO???
    

if __name__=='__main__':
    main()