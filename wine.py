import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler #для масштабирования данных

wine = load_wine()
data = pd.DataFrame(data=wine.data, columns=wine.feature_names)
data['target'] = wine.target

X = data.drop('target', axis=1)
y = data['target']

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mlp = MLPClassifier(hidden_layer_sizes=(10,), max_iter=500, random_state=42) # можно настроить параметры
mlp.fit(X_train, y_train)

from sklearn.metrics import accuracy_score
y_pred = mlp.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Точность модели MLPClassifier: {accuracy}")