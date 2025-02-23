import numpy as np

# Исходные данные
X = np.array([
    [170, 45], [165, 44], [175, 40], [160, 30], [180, 55],
    [160, 30], [170, 35], [155, 25], [170, 65], [175, 65],
    [165, 55], [180, 75], [160, 55], [175, 60], [160, 50],
    [170, 90], [165, 85], [175, 95], [160, 80], [180, 100],
    [150,75], [155, 80]
])
y = np.array([0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,2,2,2,2,2,2,2])

# Сохраняем оригинальные параметры нормализации
original_mean = X.mean(axis=0)
original_std = X.std(axis=0)
X_normalized = (X - original_mean) / original_std  # Нормализованные данные для обучения

# One-hot кодировка
def one_hot(y, num_classes):
    return np.eye(num_classes)[y]
y_one_hot = one_hot(y, 3)

# Гиперпараметры
input_size = 2
hidden_size = 16  
output_size = 3
learning_rate = 0.01
epochs = 15000  

# Инициализация весов
def he_initialization(input_size, output_size):
    return np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)

W1 = he_initialization(input_size, hidden_size)
b1 = np.zeros((1, hidden_size))
W2 = he_initialization(hidden_size, output_size)
b2 = np.zeros((1, output_size))

# Функции активации
def relu(x):
    return np.maximum(0, x)

def softmax(x):
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)

# Обучение
for epoch in range(epochs):
    # Прямое распространение
    z1 = np.dot(X_normalized, W1) + b1
    a1 = relu(z1)
    z2 = np.dot(a1, W2) + b2
    a2 = softmax(z2)
    
    # Потери
    loss = -np.sum(y_one_hot * np.log(a2 + 1e-8)) / len(X)
    
    # Обратное распространение
    dz2 = a2 - y_one_hot
    dW2 = np.dot(a1.T, dz2) / len(X)
    db2 = np.sum(dz2, axis=0, keepdims=True) / len(X)
    
    dz1 = np.dot(dz2, W2.T) * (z1 > 0)
    dW1 = np.dot(X_normalized.T, dz1) / len(X)
    db1 = np.sum(dz1, axis=0, keepdims=True) / len(X)
    
    # Обновление весов
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    # Логирование
    if epoch % 1000 == 0:
        predictions = np.argmax(a2, axis=1)
        accuracy = np.mean(predictions == y)
        print(f"Epoch {epoch}, Loss: {loss:.4f}, Accuracy: {accuracy:.2%}")

# Функция предсказания
def predict(X):
    z1 = np.dot(X, W1) + b1
    a1 = relu(z1)
    z2 = np.dot(a1, W2) + b2
    a2 = softmax(z2)
    return np.argmax(a2, axis=1)

#test
for i in range(len(X)):
    pred = predict(X[i].reshape(1, -1))
    print(f"Вход: {X[i]}, Истинный класс: {y[i]}, Предсказанный класс: {pred[0]}")

# Интерактивный режим
def predict_interactive():
    print("\nВведите данные (рост вес) или 'q' для выхода:")
    while True:
        user_input = input("> ")
        if user_input.lower() == 'q':
            break
        try:
            height, weight = map(float, user_input.split())
            inp = np.array([[height, weight]])
            normalized_inp = (inp - original_mean) / original_std
            pred = predict(normalized_inp)
            status = ["Недостаточный вес", "Нормальный вес", "Избыточный вес"][pred[0]]
            print(f"Результат: {status}")
        except:
            print("Ошибка ввода! Пример: 170 65")

predict_interactive()