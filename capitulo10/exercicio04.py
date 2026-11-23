import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, classification_report

# Carregar o conjunto de dados digits
digits = load_digits()
X, y = digits.data, digits.target

# Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Normalizar os dados
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Treinar a rede neural
clf = MLPClassifier(hidden_layer_sizes=(10,), random_state=42)
clf.fit(X_train, y_train)

# Predições
y_pred = clf.predict(X_test)

# ===============================
# METRICAS DE DESEMPENHO
# ===============================
matriz_confusao = confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))

sns.heatmap(matriz_confusao, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predição')
plt.ylabel('Real')
plt.show()

indices_errados = np.where(y_test != y_pred)[0]
num_erros = min(8, len(indices_errados))

plt.figure(figsize=(12, 4))
for i in range(num_erros):
    idx = indices_errados[i]
    plt.subplot(1, num_erros, i + 1)
    plt.imshow(X_test[idx].reshape(8, 8), cmap='gray')
    plt.title(f'Real: {y_test[idx]}\nPred: {y_pred[idx]}')
    plt.axis('off')
plt.show()
