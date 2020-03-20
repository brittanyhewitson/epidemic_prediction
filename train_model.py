#import click
import dill
import numpy as np

# import models, layers, and optimizers from keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, SimpleRNN
from keras.optimizers import SGD

# Import scikit-learn helpers
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score

# Set the random seed for splitting data
RAND_SEED = 42


def model_predict(model_name, model, X_train, y_train, X_test, y_test):
    """
    """
    # Add timing stuff here

    # Fit the model
    model.fit(X_train, y_train, batch_size=1, epochs=10, verbose=0)

    # Test the model
    outputs = model.predict(X_test)
    outputs = (outputs > 0.5)

    # Generate confusion matrix and accuracy
    cm = confusion_matrix(y_test, outputs)
    accuracy = accuracy_score(y_test, outputs)

    metrics = {
        "confusion_matrix": cm,
        'accuracy': accuracy
    }
    return metrics


def build_simple_classifier(X_train):
    """
    """
    model = Sequential()
    model.add(Dense(12, input_dim=24, activation='relu'))
    model.add(Dense(12, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def build_simple_rnn(X_train):
    """
    TODO after we figure out data
    """
    model = Sequential()
    model.add(SimpleRNN(10, input_shape=(None, 1), return_sequences=False))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


def build_lstm():
    """
    TODO after we figure out data
    """
    model = Sequential()
    # Add LSTM layer
    #model.add(LSTM(32, return_sequences=True, stateful=True, batch_input_shape))
    return model


def main():
    # Read input and output data
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
    
    simple_classifier = build_simple_classifier(X_train)
    classifier_results = model_predict(
                'simple_classifier',
                simple_classifier,
                X_train,
                y_train,
                X_test,
                y_test
            )


    # Build SGD

    # Build perceptron

    # Build Simple RNN
    # TODO: Need to understand data better before we use RNN, since the data isn't organized
    # into time series!!!!!!
    #X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    #X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
    #model = build_simple_rnn(X_train)

    # Build LSTM
    #model = build_lstm()

    print(classifier_results["confusion_matrix"])
    print(classifier_results["accuracy"])
    

if __name__=='__main__':
    main()