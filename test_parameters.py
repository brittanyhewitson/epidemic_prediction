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

def test_by_date(params=None):
    """
    """
    # Get the input data
    data_list = get_data(data_choice="by_date")    
    
    # Set up parameters
    if not params:
        params = {
            "nn_params": {
                "epochs": [150],
                "batch_size": [300]
            },
            "neighbours": [5, 10],
            "estimators": [10, 50, 100]
        }

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
                    params["nn_params"],
                    verbose=True,
                )
        # Test different learning rates and optimizers, 
        # Test MLP
        mlp = build_mlp(data["X_train"])
        mlp_results = test_nn(
                    mlp,
                    data,
                    'mlp',
                    params["nn_params"],
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
        knn_results = test_knn(data, params["neighbours"], verbose=True)
		
        # Test RF
        rf_results = test_rf(data, params["estimators"], verbose=True)

        # Visualize results
        all_results = [simple_classifier_results, [svm_results], knn_results, rf_results]			
        # Sort all results
        for result_list in all_results:
            # Sort according to accuracy
            sorted_list = sorted(result_list, key=itemgetter("accuracy"), reverse=True)
            result = sorted_list[0]
            best_results[result["name"]].append(result["accuracy"]*100)

        dates.append(data["date"])
    plot_accuracy_by_date_subplot(dates, best_results)
    return best_results


def test_nn_params(data, model_params=None, test_params=None):
    """
    """
    if not model_params:
        model_params = {
            "learning_rates": [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001],
            "optimizers": ["adam", "sgd", "rmsprop", "adagrad", "adadelta", "adamax", "nadam"] 
        }
    if not test_params:
        test_params = {
            "epochs": [1, 5, 10, 50, 100, 500, 1000, 5000],
            "batch_size": [1, 5, 10, 50, 100, 500, 1000]
        }
    
    all_mlp_results = []
    for learning_rate in model_params["learning_rates"]:
        for optimizer in model_params["optimizers"]:
            # Build the model
            mlp = build_mlp(data["X_train"], learning_rate=learning_rate, optimizer=optimizer)

            # Test the model
            mlp_results = test_nn(mlp, data, 'mlp', test_params, verbose=True)



@click.command()
@click.option("--gpu", is_flag=True)
@click.option("--data_choice", default="small_data", type=click.Choice(DATA_CHOICES))
@click.option("--show_all_plots", is_flag=True)
def main(**kwargs):
    # Set up GPU/CPU
    setup_gpu(kwargs["gpu"])

    # Test data by the date
    date_results = test_by_date()

    # Read input data
    data = get_data(data_choice=kwargs["data_choice"])

    # Test nn parameters


    # Test neighbours


    # Test estimators




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
    
    



if __name__=='__main__':
    main()