import numpy as np
import torch
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from torch import nn
import torch.optim as optim
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#загрузка датасета и его настройка 
data_path = 'flowers'

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]),
])
#/Users/starfire/Downloads/flowers

# Загрузка датасета и получение имен классов
full_dataset = datasets.ImageFolder(root=data_path, transform=transform)
class_names = full_dataset.classes  # Получаем имена классов из датасета

train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(
    full_dataset, 
    [train_size, test_size],
    generator=torch.Generator().manual_seed(42)
)

# Создание DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

#создание модели
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            # Блок 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),  
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Блок 2
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),  
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Блок 3
            nn.Conv2d(32, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16), 
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Классификатор
            nn.Flatten(),
            nn.Linear(16 * 28 * 28, 256),
            nn.BatchNorm1d(256), 
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, len(class_names))
        )
    
    def forward(self, x):
        return self.model(x)
    
#определение параметров
model = CNN().to(device)
criterion = nn.CrossEntropyLoss()  
optimizer = optim.Adam(
    model.parameters(),
    lr=0.001,
    betas=(0.9, 0.999),
    weight_decay=0.0
)
epochs = 10
best_accuracy = 0.0
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)

def train():
    global best_accuracy
    for epoch in range(epochs):
        model.train()
        total = 0
        total_loss = 0
        correct = 0
        
        for batch, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if (batch+1) % 10 == 0:
                print(f'Эпоха [{epoch+1}/{epochs}], Шаг [{batch+1}/{len(train_loader)}]')
                print(f'Потери: {total_loss/(batch+1):.4f}')
                print(f'Точность: {100 * correct/total:.2f}%')
        
        # Тестирование
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (labels == predicted).sum().item()

            val_accuracy = 100 * val_correct / val_total
            print(f'Эпоха {epoch+1}, Валидационная точность: {val_accuracy:.2f}%')
            
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                torch.save(model.state_dict(), 'best_flower.pth')
        
        scheduler.step(val_loss)

def predict(image_path, model, transform, class_names):
    # Загрузка и преобразование изображения
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    
    # Предсказание
    model.eval()
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        #predicted_prob, predicted = torch.max(probabilities, 1)
        
        # Получаем топ-3 предсказания
        top3_prob, top3_indices = torch.topk(probabilities, 3)
        
        results = []
        for prob, idx in zip(top3_prob[0], top3_indices[0]):
            results.append(f"{class_names[idx]}: {prob.item()*100:.2f}%")
        
        return results

if __name__ == '__main__':
    # Обучение модели
    train()
    predict()
    
    # Загрузка лучшей модели
    model.load_state_dict(torch.load('best_flower.pth'))
    
    # Путь к тестовому изображению
    test_image_path = 'sun1.jpg'
    
    # Получаем предсказание
    predictions = predict(test_image_path, model, transform, class_names)
    print("\nТоп-3 предсказания:")
    for pred in predictions:
        print(pred)
