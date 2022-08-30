import pandas as pd
import pickle
import traceback
import flask
from flask import Flask, request, jsonify

flask_app = Flask(__name__)

modelClassifier = pickle.load(open("model.pickle", "rb"))
score = pickle.load(open("score.pickle", "rb"))

tfidf = pickle.load(open("vectorizer.pickle", 'rb'))
multilabel = pickle.load(open("multilabel.pickle", 'rb'))

@flask_app.route("/app", methods=['GET','POST'])
def predict():
    if flask.request.method == 'GET':
        return "Prediction page"
    if flask.request.method == 'POST':
        try:

            stringToPredict = request.get_json()["Text"]
            vectorizedString = tfidf.transform([stringToPredict])
            print(modelClassifier.predict(vectorizedString).toarray())
            prediction = modelClassifier.predict(vectorizedString).toarray()
            resultat = multilabel.inverse_transform(prediction)

            return jsonify( {'prediction': str(resultat) },{ "L'accuarcy du modéle " :score})
        except:
            return jsonify({'trace': traceback.format_exc()})
if __name__ == "__main__":
    flask_app.run(debug=True)