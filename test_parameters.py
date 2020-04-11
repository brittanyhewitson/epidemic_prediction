import os
import sys
import click
import copy
import logging

import numpy as np
from operator import itemgetter

import matplotlib.pyplot as plt

from models import (
    build_simple_classifier,
    build_svm,
	build_knn,
    build_rf,
    model_predict,
)
from visualize_data import (
    plot_simple_classifier_scatter,
    plot_simple_classifier_heatmap,
    plot_simple_classifier_bar,
    plot_history,
    plot_knn_bar,
    plot_rf_bar,
    plot_all_results,
    plot_accuracy_by_date,
)
from preprocess_data import(
    get_data,
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

def setup_gpu(gpu=False):
    """
    """
    if gpu:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    else:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
        os.environ["CUDA_VISIBLE_DEVICES"] = ""


def test_simple_classifier(model, data, params, verbose=False):
    """
    """
    results = []
    for epochs in params["epochs"]:
        for batch_size in params["batch_size"]:
            test_params = {
                "epochs": epochs,
                "batch_size": batch_size
            }
            classifier_results = model_predict(
                        'simple_classifier',
                        model,
                        data,
                        verbose=False,
                        validation_data=(data["X_test"], data["y_test"]),
                        **test_params,
                    )
            if verbose:
                print("accuracy for {}: {}".format(test_params, classifier_results["accuracy"]))
            classifier_results["params"] = test_params
            results.append(classifier_results)
    return results


def test_knn(data, neighbours, verbose=False):
    """
    """
    results = []
    for n in neighbours:
        knn_classifier = build_knn(n=n)
        logging.info("Testing n_neighbours={}".format(n))
        classifier_results = model_predict(
                    'knn',
                    knn_classifier,
                    data,
                ) 
        if verbose:
            print("accuracy for n_neighbours={}: {}".format(n, classifier_results["accuracy"]))
        classifier_results["params"] = {"n_neighbours": n}
        results.append(classifier_results)
    return results

def test_rf(data, estimators, verbose=False):
    """
    """
    results = []
    for n in estimators:
        rf_classifier = build_rf(n=n)
        logging.info("Testing n_estimators={}".format(n))
        classifier_results = model_predict(
                    'rf',
                    rf_classifier,
                    data,
                ) 
        if verbose:
            print("accuracy for n_estimators={}: {}".format(n, classifier_results["accuracy"]))
        classifier_results["params"] = {"n_estimators": n}
        results.append(classifier_results)
    return results

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
    simple_params = {
        "epochs": [10],
        "batch_size": [100]
    }
    neighbours = [1, 2, 5, 10, 20, 50, 100]
    estimators = [1, 2, 5, 10, 20, 50, 100]

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
        simple_classifier_results = test_simple_classifier(
                    simple_classifier,
                    data,
                    simple_params,
                    verbose=True,
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
            #plot_simple_classifier_scatter(simple_classifier_results)
            #plot_simple_classifier_bar(simple_classifier_results)
            plot_simple_classifier_heatmap(simple_classifier_results, save_fig=True)
            plot_history(simple_classifier_results)

            # TODO: Add SVM later for big data and small data

            # KNN    
            plot_knn_bar(knn_results, save_fig=True)
            plot_rf_bar(rf_results, save_fig=True)

            # Compare Classifiers
            all_results = [simple_classifier_results, [svm_results], knn_results, rf_results]
            plot_all_results(all_results, save_fig=True)

        all_results = [simple_classifier_results, [svm_results], knn_results, rf_results]
        
        # Sort all results
        for result_list in all_results:
            # Sort according to accuracy
            sorted_list = sorted(result_list, key=itemgetter("accuracy"), reverse=True)
            result = sorted_list[0]
            best_results[result["name"]].append(result["accuracy"]*100)

        dates.append(data["date"])
    
    if len(np.unique(dates)) > 1:
        plot_accuracy_by_date(dates, best_results)
    



if __name__=='__main__':
    main()