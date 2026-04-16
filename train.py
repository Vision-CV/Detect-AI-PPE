from ultralytics import YOLO

# 1. Загружаем предобученную модель (например, YOLOv8n - Nano версия)
model = YOLO('yolov8n.pt')

# 2. Запускаем обучение
results = model.train(
    data='model\data.yaml',  # ← путь к файлу data.yaml из датасета
    epochs=100,                      # количество эпох
    imgsz=640,                       # размер входного изображения
    batch=16,                        # размер батча (зависит от GPU)
    device='cpu',                    # используем GPU
    project='ppe_detection',         # папка для результатов
    name='yolov8n_helmet_v1'         # имя эксперимента
)

# 3. Оценка на тестовой выборке
metrics = model.val()

# 4. Экспорт модели (опционально)
model.export(format='onnx') 