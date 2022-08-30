
import ast
import pickle
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

import neattext.functions as nfx
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import mlflow
from skmultilearn.problem_transform import LabelPowerset
import neattext as nt
import mlflow.sklearn

if __name__ == "__main__":
    mlflow.set_experiment(experiment_name="mlflow demo")
    df = pd.read_csv(r"C:\Users\Ahmed Guebsi\Downloads\Ahmed_repo\dataset\so_dataset_2_tags1.csv")
    from skmultilearn.problem_transform import LabelPowerset

    print(df.head())
    print(df.dtypes)
    print("les valeur Null du Dataset ")
    print(df.isnull().sum())
    print("Les nombre de text associer à chaque label")
    print(df['tags'].value_counts())
    print("description du dataset")
    print(df.describe())
    print("Les informations sur dataframe")
    print(df.info())
    print("les colones ",df.columns)
    print("nettoyage")
    df['title'].apply(lambda x:nt.TextFrame(x).noise_scan())
    df['title'].apply(lambda x:nt.TextExtractor(x).extract_stopwords())
    df['title'].apply(nfx.remove_stopwords)
    corpus = df['title'].apply(nfx.remove_stopwords)
    print(corpus)
    print(df['tags'])
    df['tags']=df['tags'].apply(lambda x:ast.literal_eval(x))
    print(df['tags'])
    print(type(df['tags']))
    multilabel = MultiLabelBinarizer()
    y= multilabel.fit_transform(df['tags'])
    print(y)


    print("les classes :")
    print(multilabel.classes_)
    tfidf = TfidfVectorizer( )
    Xfeatures = tfidf.fit_transform(corpus)

    X_train,X_test,y_train,y_test = train_test_split(Xfeatures,y,test_size=2,random_state=42)
    LP_clf = LabelPowerset(MultinomialNB())

    print("le model :",LP_clf)

    LP_clf.fit(X_train,y_train)
    br_prediction = (LP_clf.predict(X_test)).toarray()
    score=LP_clf.score(X_test,y_test)*100
    print("Score =" ,score)
    mlflow.log_metric("accuracy", score)
    mlflow.sklearn.log_model(LP_clf,"model")

