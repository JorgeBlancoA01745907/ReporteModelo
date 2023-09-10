import random
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import learning_curve

# Cargar datos
df1 = load_iris()

# Imprimir algunas muestras del dataset
print("Muestras del dataset Iris:")
for i in range(10):
    num = random.randint(1,99)
    print(f"Features: {df1.data[num]}, Etiqueta: {df1.target[num]}")

X, y = df1.data, df1.target

# Dividir datos de entrada y de salida para digits
X_1 = df1.data
y_1 = df1.target


# Dividir los datos en conjunto de entrenamiento (70%) y conjunto de prueba (30%) para digits con una semilla random
# Conjunto 1
X1_train, X1_test, y1_train, y1_test = train_test_split(X_1, y_1, test_size=0.3, random_state=random.randint(1,100))

# Pasar datos de salida esperados de las pruebas a lista
y1_test = y1_test.tolist()

# Crear un gráfico de dispersión para el conjunto de entrenamiento
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.scatter(X1_train[:, 0], X1_train[:, 1], c=y1_train, cmap=plt.cm.Set1, edgecolor='k')
plt.title("Conjunto de Entrenamiento (Iris)")
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")

# Crear un gráfico de dispersión para el conjunto de prueba
plt.subplot(1, 2, 2)
plt.scatter(X1_test[:, 0], X1_test[:, 1], c=y1_test, cmap=plt.cm.Set1, edgecolor='k')
plt.title("Conjunto de Prueba (Iris)")
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")

plt.tight_layout()
plt.show()

mejor_k = 0
mejor_acc = 0

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Probar diferentes valores de k y encontrar el mejor
for k in range(1, 10):  # Probar valores de k de 1 a 20
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X1_train, y1_train)

    predicciones = knn.predict(X1_test)
    
    accuracy = accuracy_score(y1_test, predicciones)
    print("k utilizada: ", k)
    print("Precisión del modelo con Framework:", accuracy)
   
    if accuracy > mejor_acc:
        mejor_acc = accuracy
        mejor_k = k

modelo = KNeighborsClassifier(n_neighbors=mejor_k)
modelo.fit(X1_train, y1_train)
predicciones = modelo.predict(X1_test)

accuracy = accuracy_score(y1_test, predicciones)
print(f'Exactitud del modelo: {accuracy}')

conf_matrix = confusion_matrix(y1_test, predicciones)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel('Predicciones')
plt.ylabel('Valores reales')
plt.title('Matriz de Confusión')
plt.show()

# Crear modelos k-NN con diferentes valores de k
model_low_variance = KNeighborsClassifier(n_neighbors=1)
model_medium_variance = KNeighborsClassifier(n_neighbors=5)
model_high_variance = KNeighborsClassifier(n_neighbors=30)

# Crear tamaños de conjunto de entrenamiento crecientes
train_sizes = np.linspace(0.1, 1.0, 10)

# Calcular las curvas de aprendizaje para cada modelo
train_sizes_low, train_scores_low, test_scores_low = learning_curve(model_low_variance, X, y, train_sizes=train_sizes)
train_sizes_medium, train_scores_medium, test_scores_medium = learning_curve(model_medium_variance, X, y, train_sizes=train_sizes)
train_sizes_high, train_scores_high, test_scores_high = learning_curve(model_high_variance, X, y, train_sizes=train_sizes)

# Crear gráficos de curvas de aprendizaje
plt.figure(figsize=(12, 5))

plt.subplot(1, 3, 1)
plt.plot(train_sizes_low, np.mean(train_scores_low, axis=1), 'o-', label='Entrenamiento')
plt.plot(train_sizes_low, np.mean(test_scores_low, axis=1), 'o-', label='Prueba')
plt.title('Baja Varianza (k=1)')
plt.xlabel('Tamaño del Conjunto de Entrenamiento')
plt.ylabel('Precisión')
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(train_sizes_medium, np.mean(train_scores_medium, axis=1), 'o-', label='Entrenamiento')
plt.plot(train_sizes_medium, np.mean(test_scores_medium, axis=1), 'o-', label='Prueba')
plt.title('Varianza Media (k=5)')
plt.xlabel('Tamaño del Conjunto de Entrenamiento')
plt.ylabel('Precisión')
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(train_sizes_high, np.mean(train_scores_high, axis=1), 'o-', label='Entrenamiento')
plt.plot(train_sizes_high, np.mean(test_scores_high, axis=1), 'o-', label='Prueba')
plt.title('Alta Varianza (k=30)')
plt.xlabel('Tamaño del Conjunto de Entrenamiento')
plt.ylabel('Precisión')
plt.legend()

plt.tight_layout()
plt.show()


