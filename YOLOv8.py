import cv2
import numpy as np
from ultralytics import YOLO

# Загружаем YOLO модель
yolo_model = YOLO("best50.pt")  # Укажите правильный путь к модели

# Путь к изображению
image_path = "IMAGES_-2_0.5/image41.jpg"

# Чтение изображения для получения его размера
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"Image not found: {image_path}")

img_height, img_width = image.shape[:2]

# Применяем модель к изображению
results = yolo_model(image_path)
results[0].show()

# Устанавливаем порог уверенности
conf_threshold = 0.5  # Порог уверенности

# Получаем предсказания
for result in results:
    # Извлекаем координаты и уверенность
    boxes = result.boxes.xywh  # Координаты в формате [x, y, width, height]
    confidences = result.boxes.conf  # Уверенность
    
    # Фильтруем предсказания по порогу уверенности
    filtered_predictions = []
    for i in range(len(confidences)):
        if confidences[i] >= conf_threshold:
            # Нормализация координат и размеров бокса
            norm_x = round(boxes[i][0].item() / img_width, 2)
            norm_y = round(boxes[i][1].item() / img_height, 2)
            norm_width = round(boxes[i][2].item() / img_width, 2)
            norm_height = round(boxes[i][3].item() / img_height, 2)

            # Сохраняем нормализованные значения
            filtered_predictions.append({
                "x": norm_x,
                "y": norm_y,
                "width": norm_width,
                "height": norm_height,
                "confidence": round(confidences[i].item(), 2)
            })

    print(filtered_predictions)  # Выводим нормализованные данные