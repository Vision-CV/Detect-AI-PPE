import cv2
import os
import json
import numpy 
import argparse
import time
import sys
from datetime import datetime
from ultralytics import YOLO
from typing import List, Dict, Any, Tuple, Optional

violation_last_save = {}

def load_models(pose_model_path: str = 'modelsYolo/yolov8n-pose.pt',
                ppe_model_path: str = 'modelsYolo/best.pt') -> Tuple[YOLO, YOLO]:
    """
    Загружает модели YOLO из указанных файлов.
    Возвращает кортеж (pose_model, ppe_model)
    """
    try:
        pose_model = YOLO(pose_model_path)
        ppe_model = YOLO(ppe_model_path)
        print(f"Модели загружены:\numpyose_model -> {pose_model_path}\numpype_model -> {ppe_model_path}")
        return pose_model, ppe_model
    except Exception as e:
        print(f"Ошибка загрузки моделей: {e}")
        exit(1)

def process_frame(frame : numpy.ndarray ,pose_model:YOLO,ppe_model:YOLO, conf:float = 0.5):

    """
    Прогоняет кадры через модели YOLO и возвращает их координаты боксов
    """

    pose_res = pose_model(frame, conf=conf, verbose=False)[0]
    ppe_res  = ppe_model(frame, conf=conf, verbose=False)[0]

    persons = []
    if pose_res.boxes is not None and len(pose_res.boxes) > 0:
        for box, kpts in zip(pose_res.boxes, pose_res.keypoints.data):
            kpts_arr = kpts.cpu().numpy()
            if kpts_arr.ndim == 3: kpts_arr = kpts_arr.squeeze(0)
            
            persons.append({
                "box": [int(x) for x in box.xyxy[0].tolist()],
                "conf": float(box.conf[0]),
                "keypoints": kpts_arr 
            })

    ppe_objects = []
    if ppe_res.boxes is not None and len(ppe_res.boxes) > 0:
        for box in ppe_res.boxes:

            cls_name = ppe_model.names[int(box.cls[0])]

            if cls_name.lower() in ['human', 'person']: 
                continue
            
            ppe_objects.append({
                "box": [int(x) for x in box.xyxy[0].tolist()],
                "class": ppe_model.names[int(box.cls[0])],
                "conf": float(box.conf[0])
            })

    return {"persons": persons, "ppe_objects": ppe_objects}

def calculate_iou(boxA: List[int], boxB: List[int]) -> float:
    """Вычисляет IoU двух прямоугольников [x1, y1, x2, y2]."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

def point_in_box(point, box):
    x, y = point
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2

def expand_box(box, scale=1.5):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    dw, dh = int(w * (scale-1)/2), int(h * (scale-1)/2)
    return [x1 - dw, y1 - dh, x2 + dw, y2 + dh]

def associate_ppe_smart(persons, ppe_items, iou_thresh=0.1):
    person_ppe_map = {i: [] for i in range(len(persons))}
    
    for ppe in ppe_items:
        cls = ppe.get('class', '').lower()
        box = ppe['box']
        best_person = -1
        
        if cls == 'helmet':
            # Ищем человека, у которого голова (точки 0,1,2) внутри бокса каски
            for idx, person in enumerate(persons):
                kpts = person.get('keypoints')
                if kpts is None: continue
                head_pts = [kpts[i][:2] for i in [0,1,2] if kpts[i][2] > 0.5]
                if any(point_in_box(pt, box) for pt in head_pts):
                    best_person = idx
                    break
        elif cls == 'vest':
            # Жилет должен пересекаться с торсом (точки плеч и бёдер)
            for idx, person in enumerate(persons):
                kpts = person.get('keypoints')
                if kpts is None: continue
                torso_pts = [kpts[i][:2] for i in [5,6,11,12] if kpts[i][2] > 0.5]
                if any(point_in_box(pt, box) for pt in torso_pts):
                    best_person = idx
                    break
        else:
            # Для остальных классов используем IoU с расширением
            best_iou = 0.0
            for idx, person in enumerate(persons):
                iou = calculate_iou(expand_box(box), person['box'])
                if iou > best_iou and iou > iou_thresh:
                    best_iou = iou
                    best_person = idx
        
        if best_person != -1:
            person_ppe_map[best_person].append(ppe)
    
    return person_ppe_map

def check_compliance(
    persons: List[Dict[str, Any]],
    person_ppe_map: Dict[int, List[Dict[str, Any]]],
    required_ppe: Optional[set] = None
) -> List[Dict[str, Any]]:
    
    if required_ppe is None:
        required_ppe = {'helmet', 'vest'}

    violations = []

    # Проходим по всем людям в кадре
    for person_idx, person in enumerate(persons):
        # Получаем список СИЗ, привязанных к этому человеку
        ppe_list = person_ppe_map.get(person_idx, [])

        # Извлекаем названия классов найденных СИЗ
        present_classes = {item.get('class', '').lower() for item in ppe_list}

        # Определяем, каких классов не хватает
        missing_classes = required_ppe - present_classes

        # Если есть отсутствующие СИЗ — фиксируем нарушение
        if missing_classes:
            violations.append({
                'person_idx': person_idx,
                'person_box': person.get('box', []),
                'missing': sorted(list(missing_classes)),
                'present': sorted(list(present_classes)),
                'confidence': person.get('conf', 0.0)
            })

    return violations

def save_violation(
    frame: numpy.ndarray,
    violation: Dict[str, Any],
    camera_id: str = "cam_01",
    output_dir: str = "ppeWarning",
    log_file: str = "violations_log.json",
    persons: Optional[List[Dict]] = None,
    person_ppe_map: Optional[Dict[int, List[Dict]]] = None,
) -> Optional[str]:
    
    # Проверка наличия box
    box = violation.get('person_box')
    if not box or len(box) != 4:
        print("save_violation: некорректный person_box")
        return None

    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now()
    timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S_%f")[:-3]
    missing_str = "_".join(violation.get('missing', ['unknown']))
    filename = f"{timestamp_str}_{camera_id}_missing_{missing_str}.jpg"
    filepath = os.path.join(output_dir, filename)

    # Рисуем аннотации на копии кадра
    annotated_frame = frame.copy()
    if persons is not None:
        for idx, person in enumerate(persons):
            pbox = person.get('box')
            if not pbox:
                continue
            x1, y1, x2, y2 = pbox
            color = (0, 0, 255) if idx == violation.get('person_idx') else (0, 255, 0)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated_frame, f"Person {idx}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            kpts = person.get('keypoints')
            if kpts is not None:
                for kp in kpts:
                    if len(kp) >= 3 and kp[2] > 0.5:
                        kx, ky = int(kp[0]), int(kp[1])
                        cv2.circle(annotated_frame, (kx, ky), 2, (0, 255, 255), -1)

    if person_ppe_map is not None:
        for p_idx, ppe_list in person_ppe_map.items():
            for ppe in ppe_list:
                b = ppe.get('box')
                if b:
                    x1, y1, x2, y2 = b
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 1)
                    label = f"{ppe.get('class', '?')} {ppe.get('conf', 0):.2f}"
                    cv2.putText(annotated_frame, label, (x1, y2+15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

    text = f"Violation: {', '.join(violation.get('missing', []))}"
    cv2.putText(annotated_frame, text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    success = cv2.imwrite(filepath, annotated_frame)
    if not success:
        print(f"Ошибка сохранения изображения: {filepath}")
        return None

    # Логирование
    log_entry = {
        "timestamp": timestamp.isoformat(),
        "camera_id": camera_id,
        "violation_type": missing_str,
        "person_idx": violation.get('person_idx'),
        "person_box": box,
        "confidence": violation.get('confidence'),
        "frame_path": filepath,
        "present_classes": violation.get('present', [])
    }

    try:
        if os.path.exists(log_file):
            with open(log_file, 'r', encoding='utf-8') as f:
                logs = json.load(f)
        else:
            logs = []
    except (json.JSONDecodeError, FileNotFoundError):
        logs = []

    logs.append(log_entry)

    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)

    print(f"[VIOLATION SAVED] {missing_str} | {filepath}")
    return filepath


def draw_results(frame: numpy.ndarray,
                 persons: List[Dict],
                 person_ppe_map: Dict[int, List[Dict]],
                 violations: List[Dict]) -> numpy.ndarray:
    """
    Рисует на кадре bounding boxes людей, их ключевые точки,
    рамки СИЗ и индикацию нарушений.

    Зеленый цвет — человек в полном СИЗ.
    Красный цвет — нарушитель (отсутствует хотя бы один элемент).
    """
    # Цвета (BGR)
    COLOR_OK = (0, 255, 0)        # зеленый
    COLOR_VIOLATION = (0, 0, 255) # красный
    COLOR_PPE = (255, 0, 0)       # синий для СИЗ
    COLOR_KP = (0, 255, 255)      # желтый для ключевых точек

    # Создаем копию кадра, чтобы не испортить оригинал
    vis_frame = frame.copy()

    # Множество индексов нарушителей для быстрой проверки
    violation_indices = {v['person_idx'] for v in violations}

    # Рисуем людей и ключевые точки
    for idx, person in enumerate(persons):
        box = person.get('box')
        if not box:
            continue
        x1, y1, x2, y2 = box

        # Определяем цвет рамки
        color = COLOR_VIOLATION if idx in violation_indices else COLOR_OK

        # Рамка человека
        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)

        # Подпись (ID человека и уверенность)
        label = f"Person {idx}: {person.get('conf', 0):.2f}"
        cv2.putText(vis_frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Ключевые точки (если есть)
        kpts = person.get('keypoints')
        if kpts is not None:
            for kp in kpts:
                if len(kp) >= 3 and kp[2] > 0.5:  # уверенность видимости > 0.5
                    kx, ky = int(kp[0]), int(kp[1])
                    cv2.circle(vis_frame, (kx, ky), 3, COLOR_KP, -1)

    # Рисуем СИЗ (привязанные к людям)
    for idx, ppe_list in person_ppe_map.items():
        for ppe in ppe_list:
            box = ppe.get('box')
            if box:
                x1, y1, x2, y2 = box
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), COLOR_PPE, 1)
                label = f"{ppe.get('class', '?')} {ppe.get('conf', 0):.2f}"
                cv2.putText(vis_frame, label, (x1, y2 + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_PPE, 1)

    # Отображаем информацию о нарушениях на самом кадре
    y_offset = 30
    for v in violations:
        text = f"Person {v['person_idx']} missing: {', '.join(v['missing'])}"
        cv2.putText(vis_frame, text, (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        y_offset += 25

    return vis_frame


def main():
    parser = argparse.ArgumentParser(description="Система контроля СИЗ на основе YOLOv8")
    parser.add_argument('--source', type=str, default='0',
                        help="Источник видео: '0' для веб-камеры, путь к файлу, или rtsp-ссылка")
    parser.add_argument('--pose-model', type=str, default='modelsYolo/yolov8n-pose.pt',
                        help="Путь к модели YOLOv8-Pose")
    parser.add_argument('--ppe-model', type=str, default='modelsYolo/best.pt',
                        help="Путь к модели YOLOv8 для СИЗ")
    parser.add_argument('--conf', type=float, default=0.6,
                        help="Порог уверенности для детекции")
    parser.add_argument('--iou-thresh', type=float, default=0.3,
                        help="Порог IoU для ассоциации СИЗ с человеком")
    parser.add_argument('--no-display', action='store_true',
                        help="Не показывать окно с визуализацией")
    parser.add_argument('--save-violations', action='store_true',
                        help="Сохранять кадры с нарушениями")
    parser.add_argument('--output-dir', type=str, default='ppeWarning',
                        help="Папка для сохранения кадров нарушений")
    parser.add_argument('--camera-id', type=str, default='cam01',
                        help="Идентификатор камеры для логов")
    parser.add_argument('--resize', type=int, default=None,
                        help="Изменить размер кадра по ширине (пропорционально)")

    args = parser.parse_args()

    # Определяем источник видео
    if args.source.isdigit():
        source = int(args.source)
    else:
        source = args.source

    # Загружаем модели
    print("Загрузка моделей...")
    try:
        pose_model = YOLO(args.pose_model)
        ppe_model = YOLO(args.ppe_model)
    except Exception as e:
        print(f"Ошибка загрузки моделей: {e}")
        sys.exit(1)

    # Открываем видеопоток
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Не удалось открыть источник: {source}")
        sys.exit(1)

    # Получаем параметры видео для информации
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Видео открыто: {width}x{height}, FPS: {fps:.2f}")

    # Для расчета фактического FPS
    paused = False
    start_time = time.time()
    frame_counter = 0
    DETECT_EVERY_N_FRAMES = 3

    # Для подтверждения нарушений
    CONFIRMATION_FRAMES = 3
    # Словарь: ключ - person_idx, значение - счётчик последовательных кадров с нарушением
    violation_duration = {}
    # Храним последнее обнаруженное нарушение для каждого человека (чтобы не создавать новое каждый кадр)
    last_violation_state = {}   # ключ: person_idx, значение: set(missing_classes) или None

    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("Конец видеопотока или ошибка чтения.")
                    break

                frame_counter += 1

                # Обработка кадра моделями только каждый 7-й кадр
                if frame_counter % DETECT_EVERY_N_FRAMES == 0:
                    # Изменение размера кадра (если задано)
                    if args.resize:
                        scale = args.resize / frame.shape[1]
                        new_height = int(frame.shape[0] * scale)
                        frame_resized = cv2.resize(frame, (args.resize, new_height))
                    else:
                        frame_resized = frame

                    # Обработка кадра моделями
                    frame_data = process_frame(frame_resized, pose_model, ppe_model, conf=args.conf)

                    persons = frame_data['persons']
                    ppe_objects = frame_data['ppe_objects']

                    # Ассоциация СИЗ с людьми
                    person_ppe_map = associate_ppe_smart(persons, ppe_objects, iou_thresh=args.iou_thresh)

                    # Проверка нарушений
                    violations_raw = check_compliance(persons, person_ppe_map)

                    # Обновление длительности нарушений
                    current_violated_persons = set()
                    for v in violations_raw:
                        idx = v['person_idx']
                        missing_set = frozenset(v['missing'])
                        current_violated_persons.add(idx)
                        # Увеличиваем счётчик длительности для этого человека и типа нарушения
                        # Используем кортеж (idx, missing_set) как ключ, т.к. у одного человека может быть разный набор нарушений
                        key = (idx, missing_set)
                        violation_duration[key] = violation_duration.get(key, 0) + 1

                        # Сохраняем последнее состояние нарушения для этого человека
                        last_violation_state[idx] = v

                    # Сброс счётчиков для людей, у которых нарушение исчезло
                    for key in list(violation_duration.keys()):
                        idx, missing_set = key
                        if idx not in current_violated_persons or \
                           (idx in last_violation_state and frozenset(last_violation_state[idx]['missing']) != missing_set):
                            del violation_duration[key]

                    # Проверяем, достиг ли какой-либо счётчик порога подтверждения
                    confirmed_violations = []
                    for key, duration in violation_duration.items():
                        if duration >= CONFIRMATION_FRAMES:
                            idx, missing_set = key
                            if idx in last_violation_state:
                                v = last_violation_state[idx].copy()
                                v['missing'] = sorted(list(missing_set))
                                confirmed_violations.append(v)
                                # Сбрасываем счётчик после сохранения, чтобы избежать повторных сохранений подряд
                                violation_duration[key] = 0

                    # Сохранение подтверждённых нарушений (если включено)
                    if args.save_violations and confirmed_violations:
                        for v in confirmed_violations:
                            save_violation(frame, v, camera_id=args.camera_id, output_dir=args.output_dir)

                    # Визуализация (используем подтверждённые нарушения для отображения, но можно и raw)
                    if not args.no_display:
                        vis_frame = draw_results(frame_resized, persons, person_ppe_map, confirmed_violations)

                        # Отображение FPS
                        frame_counter += 1
                        elapsed = time.time() - start_time
                        current_fps = frame_counter / elapsed if elapsed > 0 else 0
                        fps_text = f"FPS: {current_fps:.2f} (detect every {DETECT_EVERY_N_FRAMES} frames)"
                        cv2.putText(vis_frame, fps_text, (10, 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                        cv2.imshow('PPE Monitoring', vis_frame)

            # Обработка клавиш
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                paused = not paused
                state = "ПАУЗА" if paused else "ПРОДОЛЖЕНИЕ"
                print(f"Состояние: {state}")

    except KeyboardInterrupt:
        print("\nОстановка по запросу пользователя.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Ресурсы освобождены.")

if __name__ == "__main__":

    main()