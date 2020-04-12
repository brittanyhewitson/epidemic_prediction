import logging

# import models, layers, and optimizers from keras
from keras.backend import set_session
from keras.models import Sequential
from keras.layers import LSTM, Dense, SimpleRNN
from keras.optimizers import SGD, Adam, RMSprop, Adagrad, Adadelta, Adamax, Nadam

# Import scikit-learn helpers
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


def model_predict(model_name, model, data, **kwargs):
    """
    """
    # TODO: Add timing stuff here

    # Fit the model    
    history = model.fit(data["X_train"], data["y_train"], **kwargs)
        
    # Test the model
    outputs = model.predict(data["X_test"])
    outputs = (outputs > 0.5)

    # Generate confusion matrix and accuracy
    cm = confusion_matrix(data["y_test"], outputs)
    accuracy = accuracy_score(data["y_test"], outputs)

    metrics = {
        "name": model_name,
        "confusion_matrix": cm,
        "accuracy": accuracy,
        "history": history
    }
    return metrics


def build_simple_classifier(X_train):
    """
    """
    model = Sequential()
    model.add(Dense(12, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(12, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def build_mlp(X_train, learning_rate=0.001, optimizer="adam"):
    """
    """
    # Set up nodes
    hidden_nodes = 50
    output_nodes = 1

    # Set up parameters
    activation_function = "sigmoid"
    loss = "binary_crossentropy"

    # Set up optimizer
    if optimizer == "adam":
        opt = Adam(learning_rate=learning_rate)
    elif optimizer == "sgd":
        opt = SGD(learning_rate=learning_rate)
    elif optimizer == "rmsprop":
        opt = RMSprop(learning_rate=learning_rate)
    elif optimizer == "adagrad":
        opt = Adagrad(learning_rate=learning_rate)
    elif optimizer == "adadelta":
        opt = Adadelta(learning_rate=learning_rate)
    elif optimizer == "adamax":
        opt = Adamax(learning_rate=learning_rate)
    elif optimizer == "nadam":
        opt = Nadam(learning_rate=learning_rate)

    model = Sequential()
    # Hidden layers
    model.add(Dense(
        hidden_nodes,
        activation=activation_function,
        input_dim=X_train.shape[1],
        use_bias=False,
    ))
    model.add(Dense(
        2*hidden_nodes,
        activation=activation_function,
        use_bias=False,
    ))
    model.add(Dense(
        4*hidden_nodes,
        activation=activation_function,
        use_bias=False,
    ))
    model.add(Dense(
        6*hidden_nodes,
        activation=activation_function,
        use_bias=False,
    ))
    model.add(Dense(
        8*hidden_nodes,
        activation=activation_function,
        use_bias=False,
    ))

    # Output layer
    model.add(Dense(output_nodes, activation=activation_function, use_bias=False))

    # Compile the model
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

    return model


def test_nn(model, data, model_name, params, verbose=False):
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
                        model_name,
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


def build_svm(**params):
    """
    """
    return svm.SVC(**params)


def build_knn(n=1):
    """
    """
    return KNeighborsClassifier(n_neighbors=n)


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


def build_rf(n=10):
    """
    """
    return RandomForestClassifier(n_estimators=n)


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