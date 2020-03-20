import os
#import click
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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


def main():
    # Read in the final CSV
    all_data = pd.read_csv('csv_data/11_features_engineered.csv')
    print(all_data)
    # Data Visualization
    print(all_data.columns)

    # Plot the averages for each data column
    #plot_averages(all_data)

    # Look at the cities included in the data set

    # Ohter data viz for the report.....

if __name__=='__main__':
    main()