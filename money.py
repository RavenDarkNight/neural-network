import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

# Создаем класс для датасета
class SalaryDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Загрузка и подготовка данных
def prepare_data(file_path, batch_size=4):
    # Загружаем данные без заголовков
    df = pd.read_csv(file_path, header=None, names=['Index', 'Experience', 'Salary'])
    
    # Отделяем признаки (X) и целевую переменную (y)
    X = df['Experience'].values.reshape(-1, 1)
    y = df['Salary'].values.reshape(-1, 1)
    
    # Нормализуем данные
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)
    
    # Разделяем на train и test
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)
    
    # Создаем датасеты
    train_dataset = SalaryDataset(X_train, y_train)
    test_dataset = SalaryDataset(X_test, y_test)
    
    # Создаем загрузчики данных
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, scaler_X, scaler_y

# Модель
class SalaryPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.model(x)

# Функция оценки модели
def evaluate_model(model, data_loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for X, y in data_loader:
            outputs = model(X)
            loss = criterion(outputs, y)
            total_loss += loss.item()
    return total_loss / len(data_loader)

# Функция обучения
def train_model(model, train_loader, test_loader, epochs=100, lr=0.001):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)
    
    best_loss = float('inf')
    
    print("\nНачало обучения:")
    print("-" * 50)
    
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        
        for X, y in train_loader:
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        avg_test_loss = evaluate_model(model, test_loader, criterion)
        
        if avg_test_loss < best_loss:
            best_loss = avg_test_loss
            torch.save(model.state_dict(), 'best_salary_model.pth')
        
        scheduler.step(avg_test_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f'Эпоха [{epoch+1}/{epochs}]')
            print(f'Ошибка при обучении: {avg_train_loss:.4f}')
            print(f'Ошибка при тестировании: {avg_test_loss:.4f}')
            print('-' * 50)

# Функция для предсказания
def predict_salary(model, experience, scaler_X, scaler_y):
    model.eval()
    with torch.no_grad():
        # Нормализуем входные данные
        x = scaler_X.transform([[experience]])
        x = torch.FloatTensor(x)
        
        # Получаем предсказание
        pred = model(x)
        
        # Возвращаем к исходному масштабу
        pred = scaler_y.inverse_transform(pred.numpy())
        
        return pred[0][0]

if __name__ == '__main__':
    # Получаем путь к файлу
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir,'/Users/starfire/Desktop/vision/olimp/data/Salary_dataset.csv')
    print("Загрузка данных...")
    # Загружаем и подготавливаем данные
    train_loader, test_loader, scaler_X, scaler_y = prepare_data(file_path, batch_size=4)
    print("Данные успешно загружены")
    
    # Создаем и обучаем модель
    print("\nСоздание модели...")
    model = SalaryPredictor()
    print("Модель создана")
    
    # Обучаем модель
    train_model(model, train_loader, test_loader, epochs=100)
    
    # Загружаем лучшую модель
    model.load_state_dict(torch.load('best_salary_model.pth'))
    
    # Делаем предсказания
    print("\nПредсказания зарплат:")
    print("-" * 50)
    test_experiences = [1.5, 3.0, 5.0, 7.0, 10.0]
    for exp in test_experiences:
        predicted_salary = predict_salary(model, exp, scaler_X, scaler_y)
        print(f'Опыт работы: {exp:4.1f} лет -> Предсказанная зарплата: ${predicted_salary:,.2f}')
