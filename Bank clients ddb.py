

# ## Подготовка данных

# ### Описание проекта и план работы

# Цель проекта: разработать моедль, которая сможет спрогнозировать какое количество клиентов потеряет банк. (Критерием оценки модели необходимо принять F1 меру. F1_min = 0.59)
# 
# Для достижения поставленной цели был изучен датасет с данными о клиентах банка. На его базе с помощью модели RandomForestClassifire была обучена модель. После чего в модели был устранен дисбаланс, а так же улучшен показатель F1. Тестовая вывборка из датасета показала, что метрика F1 превышает минимально необходимую 0.59. 

# ### Загрузка датасета и обработка данных

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OrdinalEncoder 
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from tqdm import tqdm
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve 
import matplotlib.pyplot as plt
import numpy as np
clients_data = pd.read_csv('/datasets/Churn.csv')

clients_data .info()


# In[2]:


clients_data.head(5)


clients_data = clients_data.dropna(subset = ['Tenure'])
clients_data.info()


# Избавимся от неинформативных признаков



clients_data.drop(["RowNumber", "CustomerId", "Surname"], axis=1, inplace=True)

# Столбец 'Exited' - целевой признак

# In[5]:


data_ohe = pd.get_dummies(clients_data, drop_first=True)



target = data_ohe['Exited']
features = data_ohe.drop('Exited', axis = 1)


# Разделим данные на тестовую, вавлидационную и тренировочную.


features_train, features_valid, target_train, target_valid = train_test_split(
    features, target, test_size=0.4 , random_state=12345)
features_valid, features_test, target_valid, target_test = train_test_split(
    features_valid, target_valid, test_size=0.5 , random_state=12345)
row = (features_train, features_valid, target_train, target_valid, features_test, target_test)
for i in row:
    print(i.shape)


# ## Исследование задачи

# ### Исследуем баланс классов


class_frequency = clients_data['Exited'].value_counts(normalize=True)
class_frequency.plot(kind='bar')


# 20% от общего числа клиентов покидают банк.
# ### Изучим модели без учета дисбаланса.

# #### DecisionTreeClassifier



get_ipython().run_cell_magic('time', '', "best_model_tree = None\nbest_result = 0\nbest_result_f1 = 0\nfor depth in range(1, 15):\n    dtc = DecisionTreeClassifier(max_depth=depth, random_state=12345)\n    dtc.fit(features_train, target_train)\n    prediction = dtc.predict(features_valid)\n    acc = accuracy_score(target_valid, prediction)\n    f1 = f1_score(target_valid, prediction)\n    if acc > best_result:\n        best_model_tree = dtc\n        best_result = acc\n    if f1 > best_result_f1:\n        best_result_f1 = f1\nprint(f'Модель с лучшими гиперпараметрами: {best_model_tree}')\nprint(f'accuracy: {best_result}')\nprint('F1:', best_result_f1)")

probabilities_valid = dtc.predict_proba(features_valid)
probabilities_one_valid = probabilities_valid[:, 1]

# < напишите код здесь >
auc_roc = roc_auc_score(
    target_valid,
    probabilities_one_valid,
    multi_class="ovr",
    average="micro",
)
print(auc_roc)


# #### LogisticRegression

# In[11]:


get_ipython().run_cell_magic('time', '', "best_model_reg = None\nbest_result = 0\nbest_result_f1 = 0\nitr_qw = (100, 200, 300, 400, 500)\n\nfor itr in itr_qw:\n    lr = LogisticRegression(random_state=12345, solver='lbfgs', max_iter=itr) \n    lr.fit(features_train, target_train)\n    prediction = lr.predict(features_valid)\n    acc = accuracy_score(target_valid, prediction)\n    f1 = f1_score(target_valid, prediction)\n    if acc > best_result:\n        best_model_reg = lr\n        best_result = acc\n    if f1 > best_result_f1:\n            best_result_f1 = f1\n        \nprint(f'Модель с лучшими гиперпараметрами: {best_model_reg}')\nprint(f'accuracy: {best_result}')\nprint('F1:', best_result_f1)")


# In[12]:


probabilities_valid = lr.predict_proba(features_valid)
probabilities_one_valid = probabilities_valid[:, 1]

# < напишите код здесь >
auc_roc = roc_auc_score(
    target_valid,
    probabilities_one_valid,
    multi_class="ovr",
    average="micro",
)
print(auc_roc)


# #### RandomForestClassifier

# In[13]:


get_ipython().run_cell_magic('time', '', "best_model_forest = None\nbest_result = 0\nbest_result_f1 = 0\nfor est in range(10, 50, 10):\n    for depth in range(1, 15):\n        rfc = RandomForestClassifier(n_estimators=est, max_depth=depth, random_state=12345)\n        rfc.fit(features_train, target_train)\n        prediction = rfc.predict(features_valid)\n        acc = accuracy_score(target_valid, prediction)\n        f1 = f1_score(target_valid, prediction)\n        if acc > best_result:\n            best_model_forst = rfc\n            best_result = acc\n        if f1 > best_result_f1:\n            best_result_f1 = f1\nprint(f'Модель с лучшими гиперпараметрами: {best_model_forst}')\nprint(f'accuracy: {best_result}')\nprint('F1:', best_result_f1)")


# In[14]:


probabilities_valid = rfc.predict_proba(features_valid)
probabilities_one_valid = probabilities_valid[:, 1]

# < напишите код здесь >
auc_roc = roc_auc_score(
    target_valid,
    probabilities_one_valid,
    multi_class="ovr",
    average="micro",
)
print(auc_roc)


# ### вывод

# Изучены модели без учета дисбаланса.
# Лучшей моделью по метрике accuracy является RandomForestClassifier.



# Разделим обучающую выборку на объекты по классам.
# Определим меньший класс.
# Скопируем меньшие данные 4 раза. (отношение между классами равно 4).
# Создадим новую обучающую выборку.
# Перемешаем данные.

def upsample(features, target, repeat):
    features_zeros = features[target == 0] 
    features_ones = features[target == 1] 
    target_zeros = target[target == 0] 
    target_ones = target[target == 1]
    
    features_upsampled = pd.concat([features_zeros] + [features_ones] * repeat) 
    target_upsampled = pd.concat([target_zeros] + [target_ones] * repeat)
    
    features_upsampled, target_upsampled = shuffle( features_upsampled, target_upsampled, random_state=12345)
    return features_upsampled, target_upsampled


# 

# In[16]:


features_upsampled, target_upsampled = upsample(features_train, target_train, 4)

rfc_us = RandomForestClassifier(n_estimators=est, max_depth=depth, random_state=12345)
rfc_us.fit(features_upsampled, target_upsampled)
predicted_valid = rfc_us.predict(features_valid)

print("F1:", f1_score(target_valid, predicted_valid))


# In[17]:


probabilities_valid_rfc = rfc_us.predict_proba(features_valid)
probabilities_one_valid_rfc = probabilities_valid_rfc[:, 1]

# < напишите код здесь >
auc_roc = roc_auc_score(
    target_valid,
    probabilities_one_valid,
    multi_class="ovr",
    average="micro",
)
print(auc_roc)


# #### Устранение дисбаланса методом downsumpling

# In[18]:


# Разделим обучающую выборку на объекты по классам.
# Определим больший класс.
# Случайным образом отбросим часть от большего класса. Отбросим 3/4 (отношение между классами равно 4).
# Создадим новую обучающую выборку.
# Перемешаем данные.

def downsample(features, target, fraction): 
    features_zeros = features[target == 0] 
    features_ones = features[target == 1] 
    target_zeros = target[target == 0] 
    target_ones = target[target == 1]
    
    features_downsampled = pd.concat([features_zeros.sample(frac=fraction, random_state=12345)] + [features_ones])
    target_downsampled = pd.concat([target_zeros.sample(frac=fraction, random_state=12345)] + [target_ones])

    features_downsampled, target_downsampled = shuffle( features_downsampled, target_downsampled, random_state=12345)

    return features_downsampled, target_downsampled



features_downsampled, target_downsampled = downsample(features_train, target_train, 0.25)

rfc_ds = RandomForestClassifier(n_estimators=est, max_depth=depth, random_state=12345)
rfc_ds.fit(features_downsampled, target_downsampled)
predicted_valid = rfc_ds.predict(features_valid)

print("F1:", f1_score(target_valid, predicted_valid))


# In[22]:


probabilities_valid = rfc_ds.predict_proba(features_valid)
probabilities_one_valid = probabilities_valid[:, 1]

# < напишите код здесь >
auc_roc = roc_auc_score(
    target_valid,
    probabilities_one_valid,
    multi_class="ovr",
    average="micro",
)
print(auc_roc)


# #### Оптимальный порог классификации

# In[26]:


model = RandomForestClassifier(n_estimators=est, max_depth=depth, random_state=12345)
model.fit(features_train, target_train)
probabilities_valid = model.predict_proba(features_valid)
probabilities_one_valid = probabilities_valid[:, 1]


predicted_valid = model.predict(features_valid)
f1_best = 0
limit = 0
for threshold in np.arange(0.1, 1, 0.05):
    predicted_test = probabilities_one_valid > threshold
    f1_scr = f1_score(target_valid, predicted_valid)
    if f1_scr > f1_best:
        f1_best = f1_scr
        limit = threshold
print(f'порог = {limit}')
print(f'best F1 metrix: {f1_best}')


# #### Вывод

# Устранение дисбаланса методом upsumpling показывает наилучшую метрику F1.
# 
# F1 = 0.631
# ROC_AUC = 0.856


# ## Тестирование модели

# Протестируем модель на тестовой выборке.

# In[27]:


predicted_test = rfc_us.predict(features_test)
print("F1:", f1_score(target_test, predicted_test))


# Метрика F1 показывает себя лучше чем в выборке, на которой она обучается.
# Однако она хуже, чем в случае с выборкой, где устранен дисбаланс.
# 
# f1 = 0,61. Пороговвое значение преодалено.

# In[34]:


probabilities_test = rfc_us.predict_proba(features_test)
probabilities_one_test = probabilities_test[:, 1]

# < напишите код здесь >
auc_roc = roc_auc_score(
    target_test,
    probabilities_one_test,
    multi_class="ovr",
    average="micro",
)
print(auc_roc)


fpr, tpr, thresholds = roc_curve(target_test, probabilities_one_test)
plt.figure()
plt.plot(fpr, tpr)
plt.ylim([0.0, 1.0])
plt.xlim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC-кривая')

plt.show() 


# ## Вывод

# - Цель данного проекта - разработать моедль, которая сможет спрогнозировать какое количество клиентов потеряет банк. (Критерием оценки модели необходимо принять F1 меру. F1_min = 0.59)
# - Для достижения этой цели были решены следующие задачи:
#     1) Данные подготовлены к работе. (Датасет загружен в проект, лишние данные из него исключены, датасет разбит на три ввыборки: обучающую, валидационную и тетсовую)
#     2) Исследованы задачи проекта.  (Обнаружен дисбаланс классов в целеом признаке, определена подходящая модель обучения). Определено, что лучшей моделью по метрике accuracy является RandomForestClassifier. Accuracy данной модели составляет 0,86. F1 метрика составляет 0,58.
#     3) Устранен дисбаланс в обучающих данных. Метод upsumpling показал наилучшую метрику F1. F1 = 0.632.
#     4) Модель проверена на тестовой выборке. Тестирование модели показало, что метрика F1 = 0,585. Значение метрики вышло ниже требуемого. В связи с этим, были установлен оптимальный порог классификации для улучшения F1 метрики.
#     
# В ходе разработки проекта были подготовлены данные для работы, а именно:  исключены пропущенные данные из исходного датасета, датасет разбит на три выборки (обучающую, валидационную и тестовую), был исключен дисбаланс в аднных, а также оптимизированна F1 мера.

