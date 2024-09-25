import os

import joblib
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from data_loader import ROOT_DIR
from sklearn.metrics import make_scorer, accuracy_score, f1_score

MODEL_PATH = os.path.join(ROOT_DIR, "models")


def build_model(params=None):
    """
    Define an untrained model.

    :return: untrained model
    """

    # <ASSIGNMENT 3.5: Define a suitable model>
    #param=none allows to assign default values when no parameter is provided.
    if params is None:
        params = {
            'n_estimators': 100,                   #These are the number of trees in forest
            'max_depth': None,                     #this is the depth of the tree, here it is none so that the tree can grow deeper
            'min_samples_split': 2,                #minimum number of samples required to split the node  
        }
    model = RandomForestClassifier(random_state=42, **params)       #random state given to control the randomness of the output (to make reproducible)
    return model


def cross_validate_model(model, X_train, y_train): 
    """
    Evaluate the cross-validation performance of the model.

    :param model: untrained model
    :param X_train: pre-processed training data (numpy array)
    :param y_train: pre-processed targets for training (numpy array)
    :return: cross-validation values
    """

    # <ASSIGNMENT 3.6: Perform a 5-fold cross-validation and report a suitable metric>
    scorer = make_scorer(accuracy_score)
    scores = cross_val_score(model,X_train,y_train,cv=5,scoring=scorer)
    return scores


def train_model(model, X_train, y_train):
    """
    Train a model on the training data.

    :param X_train: pre-processed training data (numpy array)
    :param y_train: pre-processed targets for training (numpy array)
    :return: fitted model
    """

    model.fit(X_train, y_train)

    return model


def save_model(model, model_path=MODEL_PATH):
    """
    Write model to file (note, model.pkl is not committed to git).

    :param model: model object
    :param model_path: path to model (string)
    :return: model object
    """

    os.makedirs(model_path, exist_ok=True)
    model_file = os.path.join(model_path, "model.pkl")
    joblib.dump(model, model_file)

    return model


def finetune_model(model, X_train, y_train):
    """
    Perform a gridsearch to finetune model hyperparameters and refit.

    :param model: untrained model
    :param X_train: pre-processed training data (numpy array)
    :param y_train: pre-processed targets for training (numpy array)
    :return: best parameter settings
    """

    # <ASSIGNMENT 3.8: Use a grid search to finetune model parameters>
    # for fine tuning the model, we set number of trees in forest to be 5,10,15 
    #the tree depth as 5,10
    #the minimum samples to split a node as 2,5
    # here the grid search will try all the combinations of the given parameters
    
    #Tried with best parameters as n_estimators = 50 , Max_depth = None and minimum sample split = 2 but the model was overfitting.
    # Limited the max_depth to address the overfitting issue.
    param_grid={
        'n_estimators':[5,10,15],         
        'max_depth': [5,10],
        'min_samples_split':[2,5]    
    }
    grid_search = GridSearchCV(estimator=model,param_grid = param_grid,scoring='accuracy',cv=5,n_jobs=1, verbose=1 )
    grid_search.fit(X_train,y_train)
    return grid_search.best_params_


def evaluate_model(model, X_test, y_test):
    """
    Evaluate a pre-trained model on a test set.

    :param model: pre-trained model
    :param X_test: pre-processed testing data (numpy array)
    :param y_test: pre-processed targets for training (numpy array)
    :return: training and cross-validation RMSE values
    """

    y_predicted = model.predict(X_test)
    score = confusion_matrix(y_test, y_predicted)

    return score


def load_model(model_path=MODEL_PATH):
    """
    Load a model from file.

    :param model_path: path to model (string)
    :return: model object
    """
    model_file = os.path.join(model_path, "model.pkl")
    return joblib.load(model_file)


def predict(model, X):
    """
    Predict edibility from a pre-trained model.

    :param model: trained model
    :param X: sanitized and pre-processed data entry (numpy array)
    :return: predicted edibility
    """

    y_predicted = model.predict(X)

    return y_predicted
