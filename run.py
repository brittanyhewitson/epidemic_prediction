import os
import sys
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
    plot_simple_classifier_heatmap,
    plot_history,
    plot_knn_bar,
    plot_rf_bar,
    plot_all_results,
    plot_accuracy_by_date,
)
from src.preprocess_data import(
    get_data,
)
from src.templates import (
    DATA_CHOICES,
)


def main():
    # Set up GPU/CPU
    setup_gpu(gpu=False)

    # Get the input data
    data = get_data(data_choice="small_data")[0]   
    
    # Set up parameters
    nn_params = {
        "epochs": [100],
        "batch_size": [100]
    }
    neighbours = [100]
    estimators = [100]

    # Test the simple classifier parameters
    simple_classifier = build_simple_classifier(data["X_train"])
    simple_classifier_results = test_nn(
                simple_classifier,
                data,
                'simple_classifier',
                nn_params,
                verbose=False,
            )

    # Test MLP
    mlp = build_mlp(data["X_train"])
    mlp_results = test_nn(
                mlp,
                data,
                'mlp',
                nn_params,
                verbose=False
            )

    # Test SVM
    svm_classifier = build_svm()
    svm_results = model_predict(
                'svm',
                svm_classifier,
                data,
            )

    # Test KNN
    knn_results = test_knn(data, neighbours, verbose=False)
    
    # Test RF
    rf_results = test_rf(data, estimators, verbose=False)
    
    # Compare Classifiers
    all_results = [simple_classifier_results, [svm_results], knn_results, rf_results, mlp_results]
    
    # Print accuracy for each classifier
    for result in all_results:
        print("{} ACCURACY: {}".format(result[0]["name"].upper().replace("_", " "), result[0]["accuracy"]))

    # Plot results for each classifier as a bar graph
    plot_all_results(all_results, save_fig=False) 
    

if __name__=='__main__':
    main()