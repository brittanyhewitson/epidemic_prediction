import os
import sys
import click
import copy
import logging

from models import (
    get_data,
    build_simple_classifier,
    build_svm,
    build_knn,
    model_predict,
)
from visualize_data import (
    plot_simple_classifier_scatter,
    plot_simple_classifier_heatmap,
    plot_simple_classifier_bar,
    plot_history,
    plot_knn_bar,
    plot_all_results,
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


def test_simple_classifier(model, data, params):
    """
    """
    '''
    default_params = {
        "batch_size": 10,
        "epochs": 10
    }
    results = []
    for test_param, values in params.items():
        for test_constraint in values:
            #logging.info("Testing {}={}".format(test_param, test_constraint))
            test_params = copy.deepcopy(default_params)
            test_params[test_param] = test_constraint
            classifier_results = model_predict(
                        'simple_classifier',
                        model,
                        data,
                        verbose=False,
                        **test_params
                    )
            print("accuracy for {}: {}".format(test_params, classifier_results["accuracy"]))
            classifier_results["varied"] = test_param
            classifier_results["params"] = test_params
            results.append(classifier_results)
    return results
    '''
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
            print("accuracy for {}: {}".format(test_params, classifier_results["accuracy"]))
            classifier_results["params"] = test_params
            results.append(classifier_results)
    return results


def test_knn(data, neighbours):
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
        print("accuracy for n_neighbours={}: {}".format(n, classifier_results["accuracy"]))
        classifier_results["params"] = {"n_neighbours": n}
        results.append(classifier_results)
    return results


@click.command()
@click.option("--gpu", is_flag=True)
@click.option("--big_set", is_flag=True)
def main(**kwargs):
    # Set up GPU/CPU
    setup_gpu(kwargs["gpu"])

    # Get the input data
    data = get_data(big_set=kwargs["big_set"])    
    
    # Test the simple classifier parameters
    simple_params = {
        "epochs": [1000],
        "batch_size": [50]
    }
    simple_classifier = build_simple_classifier(data["X_train"])
    simple_classifier_results = test_simple_classifier(
                simple_classifier,
                data,
                simple_params,
            )

    # Test SVM
    svm_classifier = build_svm()
    svm_results = model_predict(
                'svm',
                svm_classifier,
                data,
            )

    # Test KNN
    neighbours = [1, 2, 5, 10, 20, 50, 100]
    knn_results = test_knn(data, neighbours)

    # Visualize results

    # Simple classifier
    #plot_simple_classifier_scatter(simple_classifier_results)
    plot_simple_classifier_heatmap(simple_classifier_results, save_fig=True)
    #plot_simple_classifier_bar(simple_classifier_results)
    plot_history(simple_classifier_results)
    

    # TODO: Add SVM later for big data and small data


    # KNN    
    plot_knn_bar(knn_results, save_fig=True)

    # Compare Classifiers
    all_results = [simple_classifier_results, [svm_results], knn_results]
    plot_all_results(all_results, save_fig=True)



if __name__=='__main__':
    main()