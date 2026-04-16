from ultralytics import YOLO
import cv2

model = YOLO('modelsYolo/best.pt')
results = model('test/TEST.jpg', conf=0.3)[0]  # временно снизьте порог

# Вывод всех найденных классов
for box in results.boxes:
    cls_id = int(box.cls)
    class_name = model.names[cls_id]
    conf = float(box.conf)
    print(f"Обнаружен класс: {class_name}, уверенность: {conf:.2f}")
    x1, y1, x2, y2 = box.xyxy[0].tolist()
    cv2.rectangle(results.orig_img, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
    cv2.putText(results.orig_img, f"{class_name} {conf:.2f}", (int(x1), int(y1)-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

cv2.imshow('Object detection', results.orig_img)
cv2.waitKey(0)
print(model.names)