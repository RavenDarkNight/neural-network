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

class_names = datasets.classes
print("Доступные классы:", class_names)

def train():
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
    
    best_accuracy = 0.0
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        # Создаем словарь для подсчета точности по классам
        class_correct = {classname: 0 for classname in class_names}
        class_total = {classname: 0 for classname in class_names}
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            _, predicted = torch.max(outputs.data, 1)
            
            # Подсчет точности для каждого класса
            for label, prediction in zip(labels, predicted):
                label_name = class_names[label]
                pred_name = class_names[prediction]
                if label == prediction:
                    class_correct[label_name] += 1
                class_total[label_name] += 1
            
            if (batch_idx + 1) % 10 == 0:
                print(f'\nEpoch [{epoch+1}/{epochs}], Step [{batch_idx+1}/{len(train_loader)}]')
                print(f'Loss: {running_loss/10:.2f}')
                
                # Вывод точности по каждому классу
                print("\nТочность по классам:")
                for classname in class_names:
                    if class_total[classname] > 0:
                        accuracy = 100 * class_correct[classname] / class_total[classname]
                        print(f'{classname}: {accuracy:.2f}%')
                running_loss = 0.0
        
        # Валидация
        model.eval()
        val_class_correct = {classname: 0 for classname in class_names}
        val_class_total = {classname: 0 for classname in class_names}
        val_loss = 0.0
        
        print("\nПроверка на тестовом наборе:")
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                
                # Подсчет точности для каждого класса в валидационном наборе
                for label, prediction in zip(labels, predicted):
                    label_name = class_names[label]
                    pred_name = class_names[prediction]
                    if label == prediction:
                        val_class_correct[label_name] += 1
                    val_class_total[label_name] += 1
        
        # Вывод результатов валидации по классам
        print("\nРезультаты валидации по классам:")
        total_accuracy = 0
        valid_classes = 0
        for classname in class_names:
            if val_class_total[classname] > 0:
                accuracy = 100 * val_class_correct[classname] / val_class_total[classname]
                print(f'{classname}: {accuracy:.2f}% ({val_class_correct[classname]}/{val_class_total[classname]})')
                total_accuracy += accuracy
                valid_classes += 1
        
        avg_accuracy = total_accuracy / valid_classes
        print(f'\nСредняя точность по всем классам: {avg_accuracy:.2f}%')
        
        # Сохраняем лучшую модель
        if avg_accuracy > best_accuracy:
            best_accuracy = avg_accuracy
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'Сохранена новая лучшая модель с точностью {best_accuracy:.2f}%')
        
        scheduler.step(val_loss)

# Добавляем функцию для предсказания одного изображения
def predict_image(image_path):
    model.eval()
    # Загрузка и преобразование изображения
    image = Image.open(image_path)
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        probability, prediction = torch.max(probabilities, 1)
        
        predicted_class = class_names[prediction.item()]
        confidence = probability.item() * 100
        
        print(f'\nПредсказанный класс: {predicted_class}')
        print(f'Уверенность: {confidence:.2f}%')
        
        # Вывод вероятностей для всех классов
        print('\nВероятности для всех классов:')
        probs = probabilities[0].cpu().numpy()
        for i, (classname, prob) in enumerate(zip(class_names, probs)):
            print(f'{classname}: {prob*100:.2f}%')

if __name__ == '__main__':
    train()