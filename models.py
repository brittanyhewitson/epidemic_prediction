import dill
import pandas as pd

# import models, layers, and optimizers from keras
from keras.backend import set_session
from keras.models import Sequential
from keras.layers import LSTM, Dense, SimpleRNN
from keras.optimizers import SGD

# Import scikit-learn helpers
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

# Set the random seed for splitting data
RAND_SEED = 42


def get_data(big_set=False):
    """
    """
    # Read dataset
    if big_set:
        # Read the big input data set
        all_data = pd.read_csv("csv_data/07_feature_engineering_and_cleaning.csv")
        all_data["zika_cases"] = all_data["zika_cases"].apply(cast_to_bool)
        X = all_data.drop(["location", "date", "zika_cases"], axis=1).values
        y = all_data["zika_cases"].values
    else:
        # Read the small input and output data
        with open("csv_data/X.pkl", "rb") as f:
            X = dill.load(f)
        with open("csv_data/y.pkl", "rb") as f:
            y = dill.load(f)

    # Split into training and testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y,
        random_state=RAND_SEED,
        test_size=0.2
    )

    data = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }

    return data


def cast_to_bool(zika_cases):
    """
    """
    if zika_cases > 0:
        return 1
    else:
        return 0


def model_predict(model_name, model, data, **kwargs):
    """
    """
    # TODO: Add timing stuff here

    # Fit the model
    model.fit(data["X_train"], data["y_train"], **kwargs)

    # Test the model
    outputs = model.predict(data["X_test"])
    outputs = (outputs > 0.5)

    # Generate confusion matrix and accuracy
    cm = confusion_matrix(data["y_test"], outputs)
    accuracy = accuracy_score(data["y_test"], outputs)

    metrics = {
        "name": model_name,
        "confusion_matrix": cm,
        'accuracy': accuracy
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


def build_svm():
    """
    """
    return svm.SVC()


def build_knn(n=1):
    """
    """
    return KNeighborsClassifier(n_neighbors=n)