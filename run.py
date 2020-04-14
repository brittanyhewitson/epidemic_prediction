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
    plot_all_results,
)
from src.preprocess_data import(
    get_data,
)


def main():
    # Set up GPU/CPU
    setup_gpu(gpu=False)

    # Get the input data
    data = get_data(data_choice="single_date")   
    
    # Set up parameters
    epochs = [150]
    batch_size = [300]
    dropout = True
    regularizer = 0.0001
    optimizer = "adam"
    learning_rate = 0.001
    neighbours = [100]
    estimators = [100]

    # Test the simple classifier parameters
    simple_classifier = build_simple_classifier(
                X_train=data["X_train"],
                learning_rate=learning_rate,
                optimizer=optimizer,
                regularizer=regularizer,
                dropout=dropout
            )
    simple_classifier_results = test_nn(
                model=simple_classifier,
                data=data,
                model_name='simple_classifier',
                params={"epochs": epochs, "batch_size": batch_size},
                optimizer=optimizer,
                learning_rate=learning_rate,
                regularization=regularizer,
                dropout=dropout,
                verbose=False,
            )

    # Test MLP
    mlp = build_mlp(
                X_train=data["X_train"],
                learning_rate=learning_rate,
                optimizer=optimizer,
                regularizer=regularizer,
                dropout=dropout
            )
    mlp_results = test_nn(
                model=mlp,
                data=data,
                model_name='mlp',
                params={"epochs": epochs, "batch_size": batch_size},
                optimizer=optimizer,
                learning_rate=learning_rate,
                regularization=regularizer,
                dropout=dropout,
                verbose=False,
            )

    # Test SVM
    svm_classifier = build_svm()
    svm_results = model_predict(
                model_name='svm',
                model=svm_classifier,
                data=data,
            )

    # Test KNN
    knn_results = test_knn(
                data=data, 
                neighbours=neighbours, 
                verbose=False
            )
    
    # Test RF
    rf_results = test_rf(
                data=data, 
                estimators=estimators, 
                verbose=False
            )
    
    # Compare Classifiers
    all_results = [simple_classifier_results, mlp_results, [svm_results], knn_results, rf_results]
    
    # Print accuracy for each classifier
    for result in all_results:
        print("{} ACCURACY: {}".format(result[0]["name"].upper().replace("_", " "), result[0]["accuracy"]))

    # Plot results for each classifier as a bar graph
    plot_all_results(all_results, save_fig=True) 
    

if __name__=='__main__':
    main()