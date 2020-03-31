import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import ExtraTreesClassifier


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


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def plot_averages(df, save_fig=False):
    """
    Produces a bar plot of the average value for each column in the dataframe
    Some columns may have useless averages, such as latitude and longitude, etc.

    ARGS:
        df:         (dataframe) pandas dataframe containing the data to plot
        save_fig:   (bool) flag to indicate whether to save the plot as an image
    """
    # Calculate the average for each column
    averages = df.mean(axis=0)
    
    # Create the horizontal bar plot
    plt.barh(list(averages.index), abs(averages.values))
    plt.tight_layout()
    # Save the figure if specified
    if save_fig:
        # If the output directory does not yet exist
        if not os.path.exists("output_images"):
            os.makedirs("output_images")
        plt.savefig("output_images/average_plot.png")
    # Show the bar plot and clear the figure afterwards
    plt.show()
    plt.clf()


def plot_zika_cases_vs_time(zika_cases):
    """
    """
    # Groupby date
    zika_cases = zika_cases.sort_values(by=["date"], ascending=True).dropna()
    grouped_by_date = zika_cases.groupby(["date"]).count()
    ax = grouped_by_date.zika_cases.plot()
    # TODO: Fix x-tick label spacing
    plt.xlabel("Date")
    plt.ylabel("Count")
    plt.show()
    plt.clf()

    # Group the zika data
    #zika_by_date = zika_cases.groupby(["date", "location"]).data_field.unique()
    zika_by_location = zika_cases.groupby(["location"]).data_field.unique()
    
    # Save as CSVs for inspection
    #zika_by_date.reset_index().to_csv("zika_by_date.csv", index=False)
    #zika_by_location.reset_index().to_csv("zika_by_location.csv", index=False)

    # Get a unique list of all the locations
    locations = zika_cases["location"].unique()

    # Group the zika data
    zika_by_location = zika_cases.groupby(["location"])

    # Plot all of the cases vs time for each location
    for location in locations:
        output = zika_by_location.get_group(location).sort_values(["date"], ascending=True)
        equals_zero = output["zika_cases"] == 0
        if equals_zero.all():
            continue
        plt.plot(output["date"], output["zika_cases"])
        plt.xlabel("Date")
        plt.xticks(rotation=70)
        plt.ylabel("Zika Cases")
        plt.title(location)
        plt.tight_layout()
        plt.show()
        plt.clf()


def plot_feature_output_correlation(all_data):
    """
    """
    all_data["zika_cases"] = all_data["zika_cases"].apply(cast_to_bool)
    
    # Try simple correlation
    corrmat = all_data.corr()
    sns.heatmap(corrmat, cmap=sns.color_palette("RdBu_r"), square=True, annot=True, annot_kws={"size": 6})
    plt.tight_layout()
    plt.show()
    plt.clf()
    
    # Try a tree
    X = all_data.drop(["zika_cases", "date", "location"], axis=1)
    y = all_data["zika_cases"]

    model = ExtraTreesClassifier(n_estimators=20)
    model.fit(X.values, y.values)
    plt.barh(list(X.columns), model.feature_importances_)
    plt.tight_layout()
    plt.show()
    plt.clf()


def main():
    # Read in CSV data
    #all_data = pd.read_csv('csv_data/11_features_engineered.csv')
    zika_cases = pd.read_csv("csv_data/03_infection_data_final.csv")
    all_data = pd.read_csv("csv_data/07_feature_engineering_and_cleaning.csv")

    # Data Visualization
    # Plot the averages for each data column
    plot_averages(all_data)

    # Total number of cases over time
    plot_zika_cases_vs_time(zika_cases)

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