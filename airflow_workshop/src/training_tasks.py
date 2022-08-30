import os
import pathlib
import shutil
from hyperopt import Trials, hp, STATUS_OK, tpe, fmin
import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

import ast
import pickle
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
import neattext as nt
import neattext.functions as nfx
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer






def data_validation(run_id,  **context):
    """Data validation"""

    df = pd.read_csv("so_dataset_2_tags1.csv")
    print(df.head())
    print(df.info())
    print(df.dtypes)
    print("les valeur Null du Dataset ")
    print(df.isnull().sum())
    print("Les nombre de text associer à chaque label")
    print(df['tags'].value_counts())
    print("description du dataset")
    print(df.describe())
    print("Les informations sur dataframe")
    print(df.info())
    print("les colones ", df.columns)



def data_preparation(run_id, **context):
    """Data preparation"""

    assert os.environ.get("DATA_INTERMEDIA_FOLDER")
    run_path = pathlib.Path(os.environ.get("DATA_INTERMEDIA_FOLDER"), run_id)
    input_path = pathlib.Path(
        run_path,
        "so_dataset_2_tags1.csv",
    )
    df = pd.read_csv(input_path)
    print("nettoyage")
    df['title'].apply(lambda x: nt.TextFrame(x).noise_scan())
    df['title'].apply(lambda x: nt.TextExtractor(x).extract_stopwords())
    df['title'].apply(nfx.remove_stopwords)
    corpus = df['title'].apply(nfx.remove_stopwords)
    print(corpus)
    print(df['tags'])
    df['tags'] = df['tags'].apply(lambda x: ast.literal_eval(x))
    print(df['tags'])
    print(type(df['tags']))
    multilabel = MultiLabelBinarizer()
    y = multilabel.fit_transform(df['tags'])
    print(y)

    print("les classes :")
    print(multilabel.classes_)
    tfidf = TfidfVectorizer()
    Xfeatures = tfidf.fit_transform(corpus)

    X_train, X_test, y_train, y_test = train_test_split(Xfeatures, y, test_size=2, random_state=42)
    LP_clf = LabelPowerset(MultinomialNB())

    print("le model :", LP_clf)





def model_training(run_id,**context):
    """Model training"""
    assert os.environ.get("DATA_INTERMEDIA_FOLDER")
    df = pd.read_csv("so_dataset_2_tags1.csv")
    multilabel = MultiLabelBinarizer()
    y = multilabel.fit_transform(df['tags'])
    print(y)

    print("les classes :")
    print(multilabel.classes_)
    tfidf = TfidfVectorizer()
    Xfeatures = tfidf.fit_transform(corpus)

    X_train, X_test, y_train, y_test = train_test_split(Xfeatures, y, test_size=2, random_state=42)
    LP_clf = LabelPowerset(MultinomialNB())

    print("le model :", LP_clf)

    LP_clf = LabelPowerset(MultinomialNB())
    LP_clf.fit(X_train, y_train)
    br_prediction = (LP_clf.predict(X_test)).toarray()
    score = LP_clf.score(X_test, y_test) * 100
    print("Score =", score)

    # Save model as "random_forest_model.sav"
    pickle.dump(LP_clf, open("model.pickle", "wb"))
    pickle.dump(tfidf, open("vectorizer.pickle", 'wb'))
    pickle.dump(multilabel, open("multilabel.pickle", "wb"))
    pickle.dump(score, open("score.pickle", "wb"))


def model_evaluation(run_id, **context):
    """Model evaluation"""
    assert os.environ.get("DATA_INTERMEDIA_FOLDER")
    run_path = pathlib.Path(os.environ.get("DATA_INTERMEDIA_FOLDER"), run_id)
    modelClassifier = pickle.load(open("model.pickle", "rb"))
    score = pickle.load(open("score.pickle", "rb"))

    tfidf = pickle.load(open("vectorizer.pickle", 'rb'))
    multilabel = pickle.load(open("multilabel.pickle", 'rb'))

    # Some evaluation for random forest model using cross validation.
    lp_eval = cross_val_score(estimator=modelClassifier, X=X_train, y=y_train, cv=10)
    print(f"Random forest model: {lp_eval.mean()}")


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


def model_validation(run_id, model_tracking, **context):
    """Model validation"""
    assert os.environ.get("DATA_INTERMEDIA_FOLDER")
    run_path = pathlib.Path(os.environ.get("DATA_INTERMEDIA_FOLDER"), run_id)

    modelClassifier = pickle.load(open("model.pickle", "rb"))
    score = pickle.load(open("score.pickle", "rb"))

    tfidf = pickle.load(open("vectorizer.pickle", 'rb'))
    multilabel = pickle.load(open("multilabel.pickle", 'rb'))

    pred_rfc = modelClassifier.predict(X_test)

    # Print classification report
    print("\n" + classification_report(y_test, pred_rfc))

    with mlflow.start_run(run_name=run_id):
        (rmse, mae, r2) = eval_metrics(y_test, pred_rfc)
        mlflow.log_params(modelClassifier.get_params())
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)
        mlflow.sklearn.log_model(
            sk_model=modelClassifier,
            artifact_path=model_tracking["artifact_path"],
            registered_model_name=model_tracking["registered_model_name"],
        )
