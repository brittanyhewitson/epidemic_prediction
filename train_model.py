#import click
import dill
import numpy as np
import matplotlib.pyplot as plt

# import models, layers, and optimizers from keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, SimpleRNN, Dropout
from keras.optimizers import SGD

#import regularization tools
from keras import regularizers
from tensorflow import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint
callbacks = [EarlyStopping(monitor='val_loss', patience=2, verbose=1), 
                               ModelCheckpoint('best_model_so_far.h5',
                               monitor='val_loss', save_best_only=True, verbose=1)]

# Import scikit-learn helpers
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

# Set the random seed for splitting data
RAND_SEED = 42

from timeit import default_timer as timer

def model_predict(model_name, model, X_train, y_train, X_test, y_test, **kwargs):
    """
    """
    # TODO: Add timing stuff here
    #timer
    
    # Fit the model
    history = model.fit(X_train, y_train, **kwargs)

    # Test the model
    outputs = model.predict(X_test)
    outputs = (outputs > 0.5)

    # Generate confusion matrix and accuracy
    cm = confusion_matrix(y_test, outputs)
    accuracy = accuracy_score(y_test, outputs)

    metrics = {
        "name": model_name,
        "confusion_matrix": cm,
        'accuracy': accuracy,
    }
    return metrics, history


def build_simple_classifier(X_train):
    """
    """
    model = Sequential()
    model.add(Dense(12, input_dim=X_train.shape[1], activation='relu', kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l1(0.02)))
    #model.add(Dropout(0.2))
    model.add(Dense(12, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def build_simple_rnn(X_train):
    """
    TODO after we figure out data
    """
    model = Sequential()
    model.add(SimpleRNN(10, input_shape=(None, 24), return_sequences=False))
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
    start_t = timer()
    # Simple classifier
    simple_classifier = build_simple_classifier(X_train)
    classifier_results, history = model_predict(
                'simple_classifier',
                simple_classifier,
                X_train,
                y_train,
                X_test,
                y_test,
                batch_size=100,
                epochs=100,
				#callbacks=callbacks,
                validation_split=0.2
				
            )

    # # SVM
    # svm_classifier = svm.SVC()
    # svm_results = model_predict(
                # 'svm',
                # svm_classifier,
                # X_train,
                # y_train,
                # X_test,
                # y_test
            # )

    # # KNN
    # knn_classifier = KNeighborsClassifier(n_neighbors = 1)
    # #knn_score=cross_val_score(knn_classifier,X,y,cv=10)
    # knn_results = model_predict(
                # 'knn',
                # knn_classifier,
                # X_train,
                # y_train,
                # X_test,
                # y_test
            # )   
    
	# Build SGD?

    # Build Simple RNN
    # TODO: Need to understand data better before we use RNN, since the data isn't organized
    # into time series!!!!!!
    #X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    #X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
    #model = build_simple_rnn(X_train)

    # Build LSTM
    #model = build_lstm()
    #stop timer and display
    end_t = timer()

    # Display results
    print("------------------------------------------------")
    print("------------------------------------------------")
    print("RESULTS")
    print("------------------------------------------------")
    print("------------------------------------------------")
    print("SIMPLE CLASSIFIER:")
    print(classifier_results["confusion_matrix"])
    print("accuracy: {}".format(classifier_results["accuracy"]))
    print("------------------------------------------------")
    print("------------------------------------------------")
    # print("SVM:")	
    # print(svm_results["confusion_matrix"])
    # print("accuracy: {}".format(svm_results["accuracy"]))
    # print("------------------------------------------------")
    # print("------------------------------------------------")
    # print("KNN:")
    # print(knn_results["confusion_matrix"])
    # print("accuracy: {}".format(knn_results["accuracy"]))
    # print("------------------------------------------------")
    # print("elapsed_timer = {} seconds".format(end_t-start_t))
    # print("------------------------------------------------")

	#plot
    plt.plot(range(1, (len(history.history['accuracy'])+1)), history.history['accuracy'])
    plt.plot(range(1, (len(history.history['val_accuracy'])+1)), history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    

if __name__=='__main__':
    main()