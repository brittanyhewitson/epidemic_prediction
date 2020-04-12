import os
import sys
import click
import copy
import logging

import numpy as np
from operator import itemgetter

import matplotlib.pyplot as plt

from src.helpers import (
    setup_gpu,
)
from src.models import (
    build_simple_classifier,
    build_svm,
    build_knn,
    build_mlp,
    build_rf,
    model_predict,
    test_nn,
    test_knn,
    test_rf,
)
from src.visualize_data import (
    plot_nn_heatmap,
    plot_history,
    plot_knn_bar,
    plot_rf_bar,
    plot_all_results,
    plot_accuracy_by_date_subplot,
)
from src.preprocess_data import(
    get_data,
)
from src.templates import (
    DATA_CHOICES,
)

'''
# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", 
    stream=sys.stderr, 
    level=logging.INFO
)
'''

@click.command()
@click.option("--gpu", is_flag=True)
@click.option("--data_choice", default="small_data", type=click.Choice(DATA_CHOICES))
@click.option("--show_all_plots", is_flag=True)
def main(**kwargs):
    # Set up GPU/CPU
    setup_gpu(kwargs["gpu"])

    # Get the input data
    data_list = get_data(data_choice=kwargs["data_choice"])    
    
    # Set up parameters
    nn_params = {
        "epochs": [10],
        "batch_size": [100]
    }
    neighbours = [10]
    estimators = [10]

    # Set up empty variables
    best_results = {
        "simple_classifier": [],
        "svm": [],
        "knn": [],
        "rf": []
    }
    
    dates = []
    for data in data_list:
        # If all the training data for this date is the same, then don't use this date
        if len(np.unique(data["y_train"])) == 1:
            continue

        # Test the simple classifier parameters
        simple_classifier = build_simple_classifier(data["X_train"])
        simple_classifier_results = test_nn(
                    simple_classifier,
                    data,
                    'simple_classifier',
                    nn_params,
                    verbose=True,
                )

        # Test MLP
        mlp = build_mlp(data["X_train"])
        mlp_results = test_nn(
                    mlp,
                    data,
                    'mlp',
                    nn_params,
                    verbose=True
                )

        # Test SVM
        svm_classifier = build_svm()
        svm_results = model_predict(
                    'svm',
                    svm_classifier,
                    data,
                )

        # Test KNN
        knn_results = test_knn(data, neighbours, verbose=True)
		
        # Test RF
        rf_results = test_rf(data, estimators, verbose=True)

        # Visualize results
        if kwargs["show_all_plots"]:
            # Simple classifier
            plot_nn_heatmap(simple_classifier_results, save_fig=False)
            plot_history(simple_classifier_results)

            # MLP
            plot_nn_heatmap(mlp_results, save_fig=False)
            plot_history(mlp_results)

            # TODO: Add SVM later for big data and small data

            # KNN    
            plot_knn_bar(knn_results, save_fig=False)

            # RF
            plot_rf_bar(rf_results, save_fig=True)
			
            # Compare Classifiers
            all_results = [simple_classifier_results, [svm_results], knn_results, rf_results, mlp_results]
            plot_all_results(all_results, save_fig=False) 

        all_results = [simple_classifier_results, [svm_results], knn_results, rf_results]			
        # Sort all results
        for result_list in all_results:
            # Sort according to accuracy
            sorted_list = sorted(result_list, key=itemgetter("accuracy"), reverse=True)
            result = sorted_list[0]
            best_results[result["name"]].append(result["accuracy"]*100)

        dates.append(data["date"])
    
    if len(np.unique(dates)) > 1:
        plot_accuracy_by_date_subplot(dates, best_results)
    



if __name__=='__main__':
    main()