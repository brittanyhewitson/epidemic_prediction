#import click
import dill
import numpy as np

# Use keras to implement our own
#from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, accuracy_score

# Test some premade models for right now
from sklearn.linear_model import SGDClassifier, RidgeClassifier, Perceptron


rand_seed = 42

def model_predict(model_name, model, X_train, y_train, X_test, y_test):
    """
    """
    # Add timing stuff here
    print(X_train.shape)
    '''
    pipeline = Pipeline([
                ('features', FeatureUnion([
                    ('c1', Pipeline([
                    ('text1', ExtractText1()),
                    ('tf_idf1', TfidfVectorizer())
                    ])),
                    ('c2', Pipeline([
                    ('text2', ExtractText2()),
                    ('tf_idf2', TfidfVectorizer())
                    ]))
                ])),
                ('classifier', MultinomialNB())
                ])
    '''
    model.fit(X_train, y_train)

    model_predict = (X_test)
    model_predict = (model_predict > 0.5).astype(int)
    #accuracy = accuracy_score(y_test, model_predict)
    output = {
        'output': model_name,
        'model_predict': model_predict,
        'y_test': y_test,
        #'accuracy': accuracy
    }

    return output


def main():
    pass

    # Read input and output data
    with open("csv_data/X.pkl", "rb") as f:
        X = dill.load(f)
    with open("csv_data/y.pkl", "rb") as f:
        y = dill.load(f)

    # Split into training and testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y,
        random_state=rand_seed,
        test_size=0.2
    )

    print(X.shape)
    raise Exception()
    # Build SGD
    sgd = SGDClassifier(alpha=0.0001, max_iter=100, penalty="elasticnet")
    sgd_results = model_predict(
                    "sgd",
                    sgd,
                    X_train,
                    y_train,
                    X_test,
                    y_test
                )
    #print(sgd_results)

    # Build perceptron
    ridge = RidgeClassifier(tol=1e-2, solver="sag")
    ridge_results = model_predict(
                    "ridge",
                    ridge,
                    X_train,
                    y_train,
                    X_test,
                    y_test
                )
    #print(ridge_results)

    # Build LSTM





if __name__=='__main__':
    main()