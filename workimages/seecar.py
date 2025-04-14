import cv2
import numpy as np
import easyocr

def load_image(img_path):
    # Загрузка изображения
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Не удалось загрузить изображение по пути: {img_path}")
    
    # Преобразование в оттенки серого
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Применение билатерального фильтра для уменьшения шума
    # Параметры: 11 - размер окна, 15 - цветовая сигма, 0 - пространственная сигма
    gray = cv2.bilateralFilter(gray, 11, 15, 0)
    
    # Определение краев с помощью алгоритма Канни
    # 100 - нижний порог, 200 - верхний порог
    edges = cv2.Canny(gray, 100, 200)
    
    return img, gray, edges

def detect_plate(gray, model):
    """
    Обнаружение номерных знаков с помощью каскадного классификатора
    """
    # scaleFactor=1.1 - насколько уменьшается изображение на каждом шаге
    # minNeighbors=4 - минимальное количество соседей для подтверждения обнаружения
    plates = model.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)
    return plates


def process_plate(img, plates):

    # Обработка каждого найденного номера
    for (x, y, w, h) in plates:
        # Добавляем отступы для лучшего захвата номера
        padding = 10
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(img.shape[1], x + w + padding)
        y2 = min(img.shape[0], y + h + padding)
        
        # Рисуем прямоугольник вокруг номера
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return img

def resize_image(img, scale_percent):
    """
    Изменение размера изображения
    """
    width = int(img.shape[1] * scale_percent / 100) #img.shape[1] ширина 
    height = int(img.shape[0] * scale_percent / 100) #img.shape[0] высота
    dim = (width, height)
    return cv2.resize(img, dim, interpolation=cv2.INTER_AREA) #необходимо пересчитать пиксели cv2.INTER_AREA — лучший для уменьшения изображения (сохраняет чёткость).

def main():
    # Пути к файлам
    img_path = '/Users/starfire/Desktop/vision/workimages/images/bmw.jpg'
    model_path = '/Users/starfire/Desktop/vision/workimages/xml/cars.xml'
    
    try:
        # Загрузка каскадного классификатора
        plate_cascade = cv2.CascadeClassifier(model_path)
        if plate_cascade.empty():
            raise FileNotFoundError("Не удалось загрузить модель классификатора")
        
        # Загрузка и обработка изображения
        img, gray, edges = load_image(img_path)
        
        # Обнаружение номеров
        plates = detect_plate(gray, plate_cascade)
        
        if len(plates) == 0:
            print("Номера не обнаружены")
            return
        
        # Обработка и отрисовка результатов
        result_img = process_plate(img, plates)
        resized_img = resize_image(result_img, 150)  # Увеличение на 50%
        
        # Показ результата
        cv2.imshow('Номера машин', resized_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"Ошибка: {e}")

if __name__ == '__main__':
    main()

