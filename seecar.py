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
    # Инициализация EasyOCR
    reader = easyocr.Reader(['en'])
    
    # Обработка каждого найденного номера
    for (x, y, w, h) in plates:
        # Рисуем прямоугольник вокруг номера
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        # Вырезаем область с номером
        plate_region = img[y:y+h, x:x+w]
        result = reader.readtext(plate_region)
        
        if result is not None:
            text = result[0][1]
            # Добавляем распознанный текст
            cv2.putText(img, text, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2) #для наложения текста на изображение
        else:
            print('Ошибка при считывании текста')
    
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
    img_path = '/Users/starfire/Desktop/vision/images/number.jpg'
    model_path = '/Users/starfire/Desktop/vision/ML/cars.xml'
    
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
        cv2.imshow('Detected License Plates', resized_img)
        cv2.waitKey(0)

    except Exception as e:
        print(f"Ошибка: {e}")

if __name__ == '__main__':
    main()