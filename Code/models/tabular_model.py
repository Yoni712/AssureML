# models/tabular_model.py

import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

class TabularModel:
    def __init__(self):
        data = load_iris()
        self.X, self.y = data.data, data.target

        # Train a simple model
        self.model = RandomForestClassifier(random_state=42)
        self.model.fit(self.X, self.y)

        print("Tabular model trained (Iris dataset)")

    def predict_proba(self, X):
        return self.model.predict_proba(X)
