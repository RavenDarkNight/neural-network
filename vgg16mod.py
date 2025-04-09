import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader, random_split

# Оптимизированные преобразования
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),  # Добавляем аугментацию
    transforms.RandomRotation(10),      # Добавляем аугментацию
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Путь к папке с данными
data_dir = 'datasets'


# Загружаем весь датасет
dataset = datasets.ImageFolder(root=data_dir, transform=transform)

# Разделяем на train и test (70% train, 30% test)
train_size = int(0.7 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Создаем загрузчики данных
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class ANIMAL(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # Используем предобученную VGG16 для лучших результатов
        self.feature_extractor = models.vgg16(pretrained=True)
        
        # Замораживаем веса предобученной модели
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        #классификатор, который добавлен поверх VGG16   
        self.classifier = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )
        
    def forward(self, x):
        x = self.feature_extractor.features(x)  # Извлекаем признаки
        x = torch.flatten(x, 1)                # "Выравниваем" признаки в вектор
        x = self.classifier(x)                 # Классифицируем признаки
        return x
    
# Инициализация модели и перемещение на GPU если доступно
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ANIMAL().to(device)
epochs = 5 

def train():
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=0.001)  # Оптимизируем только классификатор
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3) #Если метрика не улучшается в течение 3 эпох, скорость обучения уменьшается
    
    best_accuracy = 0.0
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            #перемещаем данные либо на cpu либо на cuda
            images, labels = images.to(device), labels.to(device)

            #обучение
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            #статистика обучения
            running_loss += loss.item() #Чтобы накопить общую сумму потерь за все батчи
            _, predicted = torch.max(outputs.data, 1)   # Находит предсказанные классы для каждого примера в батче.
            total += labels.size(0)#Увеличивает общее количество обработанных примеров на размер текущего батча.
            correct += (predicted == labels).sum().item()#predicted == labels: создает тензор булевых значений (True/False), где True соответствует правильным предсказаниям.
            
            if (batch_idx + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Step [{batch_idx+1}/{len(train_loader)}], '
                      f'Loss: {running_loss/10:.2f} '
                      f'Accuracy: {100 * correct/total:.2f}%')
                running_loss = 0.0

        # тест
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
                val_correct += (predicted == labels).sum().item()
        
        val_accuracy = 100 * val_correct / val_total
        print(f'Validation Accuracy: {val_accuracy:.2f}%')
        
        # Сохраняем лучшую модель
        if val_accuracy >  best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), 'best_model.pth')
        
        scheduler.step(val_loss)
        
if __name__ == '__main__':
    train()
