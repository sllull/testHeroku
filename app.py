from flask import Flask
from flask_restful import Api, Resource
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)
api = Api(app)


class HelloWorld(Resource):
    def get(self):
        return {'hello': ''}


class Todo(Resource):
    def get(self):
        df = pd.read_csv('heart.csv', delimiter=',')
        print(df.describe())

        df = df.sample(frac=1).reset_index(drop=True)

        msk = np.random.rand(len(df)) < 0.8

        train_X = df[msk].values[:, :-1]
        train_y = df[msk].values[:, -1]
        test_X = df[~msk].values[:, :-1]
        test_y = df[~msk].values[:, -1]
        rf = RandomForestClassifier(verbose=0, n_estimators=100)
        rf.fit(train_X, train_y)

        predictions = [0] * test_X.shape[0]

        for i in range(test_X.shape[0]):
            predictions[i] = rf.predict(test_X[i, :].reshape(1, -1))[0]
        test = [["37",   "1",   "2", "130", "250",   "0",   "1", "187",   "0",   "3",   "0",   "0",   "2"]]
        pred = rf.predict(test)[0]
        print(pred)
        return {'hello': '2'}


api.add_resource(HelloWorld, '/hello')
api.add_resource(Todo, '/todo')

if __name__ == '__main__':
    app.run(debug=True)
