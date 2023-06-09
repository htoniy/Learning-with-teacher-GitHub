# Описание проекта
- Цель данного проекта - разработать моедль, которая сможет спрогнозировать какое количество клиентов потеряет банк. (Критерием оценки модели необходимо принять F1 меру. F1_min = 0.59)
- Для достижения этой цели были решены следующие задачи:
    1) Данные подготовлены к работе. (Датасет загружен в проект, лишние данные из него исключены, датасет разбит на три ввыборки: обучающую, валидационную и тетсовую)
    2) Исследованы задачи проекта.  (Обнаружен дисбаланс классов в целеом признаке, определена подходящая модель обучения). Определено, что лучшей моделью по метрике accuracy является RandomForestClassifier. Accuracy данной модели составляет 0,86. F1 метрика составляет 0,58.
    3) Устранен дисбаланс в обучающих данных. Метод upsumpling показал наилучшую метрику F1. F1 = 0.632.
    4) Модель проверена на тестовой выборке. Тестирование модели показало, что метрика F1 = 0,585. Значение метрики вышло ниже требуемого. В связи с этим, были установлен оптимальный порог классификации для улучшения F1 метрики.
    
В ходе разработки проекта были подготовлены данные для работы, а именно:  исключены пропущенные данные из исходного датасета, датасет разбит на три выборки (обучающую, валидационную и тестовую), был исключен дисбаланс в аднных, а также оптимизированна F1 мера.
