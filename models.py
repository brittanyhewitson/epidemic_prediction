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


def build_svm(**params):
    """
    """
    return svm.SVC(**params)


def build_knn(n=1):
    """
    """
    return KNeighborsClassifier(n_neighbors=n)

def build_rf(n=10):
    """
    """
    return RandomForestClassifier(n_estimators=n)