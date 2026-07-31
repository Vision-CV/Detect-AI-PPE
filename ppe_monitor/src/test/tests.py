"""
На данный момент это легаси тесты в скором времени это будет исправлено

"""
import cv2


def enhance_contrast(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl,a,b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)



def monitor_ppe_binding(persons: list, ppe_objects: list, person_ppe_map: dict, required_ppe: set = {'helmet', 'vest'}):
    """
    Комплексный мониторинг:
    1. Статистика по кадру (всего найдено vs привязано).
    2. Список НЕПРИВЯЗАННЫХ объектов (для отладки ложных срабатываний или ошибок ассоциации).
    3. Таблица статуса каждого человека.
    """
    if not persons and not ppe_objects:
        return

    # --- Шаг 1: Идентификация привязанных объектов ---
    # Создаем множество ID объектов, которые успешно привязались к людям
    assigned_obj_ids = set()
    for idx, ppe_list in person_ppe_map.items():
        for item in ppe_list:
            # Используем id() объекта в памяти как уникальный ключ
            assigned_obj_ids.add(id(item))

    # --- Шаг 2: Разделение объектов на привязанные и свободные ---
    unassigned_objects = []
    stats = {
        'helmet': {'total': 0, 'assigned': 0},
        'vest': {'total': 0, 'assigned': 0},
        'gloves': {'total': 0, 'assigned': 0},
        'boots': {'total': 0, 'assigned': 0}
    }

    for obj in ppe_objects:
        cls_name = obj['class'].lower()
        if cls_name in stats:
            stats[cls_name]['total'] += 1
            
            if id(obj) in assigned_obj_ids:
                stats[cls_name]['assigned'] += 1
            else:
                unassigned_objects.append(obj)

    # --- Шаг 3: Вывод статистики кадра ---
    print("\n" + "="*90)
    print(f"📊 FRAME STATISTICS | Persons: {len(persons)} | Total PPE Detected: {len(ppe_objects)}")
    print("-"*90)
    
    stat_header = f"{'Class':<10} | {'Total':<8} | {'Assigned':<10} | {'Unassigned':<12}"
    print(stat_header)
    print("-"*90)
    
    for cls in stats.keys():
        total = stats[cls]['total']
        assigned = stats[cls]['assigned']
        unassigned = total - assigned
        
        # Подсветка проблемных зон
        color_start = "\033[93m" if unassigned > 0 else ""
        color_end = "\033[0m"
        
        print(f"{cls:<10} | {total:<8} | {assigned:<10} | {color_start}{unassigned}{color_end}")
        
    print("="*90)

    # --- Шаг 4: Вывод НЕПРИВЯЗАННЫХ объектов (Отладка) ---
    if unassigned_objects:
        print(f"\n⚠️  UNASSIGNED OBJECTS ({len(unassigned_objects)} items found):")
        print("-"*90)
        print(f"{'ID':<6} | {'Class':<10} | {'Conf':<6} | {'Box (x1,y1,x2,y2)':<25} | {'Reason Hint'}")
        print("-"*90)
        
        for i, obj in enumerate(unassigned_objects):
            box = obj['box']
            # Простая эвристика для подсказки причины
            hint = "Check overlap/size"
            if obj['conf'] < 0.5:
                hint = "Low confidence"
            
            print(f"{i:<6} | {obj['class']:<10} | {obj['conf']:.2f}   | {str(box):<25} | {hint}")
        print("-"*90)

    # --- Шаг 5: Детальная таблица по людям ---
    if persons:
        print(f"\n👤 PERSON STATUS DETAILS:")
        print("-"*90)
        header = f"{'TrackID':<8} | {'Status':<12} | {'Helmet':<8} | {'Vest':<8} | {'Gloves':<8} | {'Boots':<8}"
        print(header)
        print("-"*90)

        for idx, person in enumerate(persons):
            track_id = person.get('track_id', 'N/A')
            ppe_list = person_ppe_map.get(idx, [])
            
            present_classes = {}
            for item in ppe_list:
                cls_name = item['class'].lower()
                conf = item['conf']
                if cls_name not in present_classes or conf > present_classes[cls_name]:
                    present_classes[cls_name] = conf

            missing = required_ppe - set(present_classes.keys())
            
            if not missing:
                status = "✅ OK"
                status_color = "\033[92m" 
            else:
                status = "❌ VIOLATION"
                status_color = "\033[91m" 
                
            reset_color = "\033[0m"

            def get_val(cls_name):
                return f"{present_classes.get(cls_name, 0):.2f}" if cls_name in present_classes else "-"

            print(f"{status_color}{str(track_id):<8} | {status:<12} | {get_val('helmet'):<8} | {get_val('vest'):<8} | {get_val('gloves'):<8} | {get_val('boots'):<8}{reset_color}")
        
        print("-"*90)
    
    print("")

def missing_str_convert_to_index(missing_str:str) -> int:
    
    if missing_str == "vest" :
        index_of_missing = 0
    elif missing_str == "helmet":
        index_of_missing = 1
    elif missing_str == "helmet_vest":
        index_of_missing = 2

    return index_of_missing
