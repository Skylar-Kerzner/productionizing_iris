import pickle
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def train_and_save_model():
    iris = load_iris()
    X_train, _, y_train, _ = train_test_split(iris.data, iris.target, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)

def load_model():
    train_and_save_model()
    with open("model.pkl", "rb") as f:
        return pickle.load(f)