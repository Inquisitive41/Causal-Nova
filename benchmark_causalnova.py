import numpy as np
import time
from CausalNova import CausalNova
from sklearn.metrics import precision_score, recall_score, f1_score


# Функция для генерации синтетических данных с известной структурой
def generate_data(
    n_samples=1000, n_vars=5, noise=0.1, nan_ratio=0.0, categorical=False
):
    np.random.seed(42)
    data = np.random.rand(n_samples, n_vars)
    # Вводим причинную связь: X0 -> X4
    data[:, 4] = 2 * data[:, 0] + np.random.rand(n_samples) * noise
    # Добавляем NaN
    if nan_ratio > 0:
        nan_mask = np.random.rand(*data.shape) < nan_ratio
        data[nan_mask] = np.nan
    # Добавляем категориальный признак
    if categorical:
        cat = np.random.choice(["A", "B", "C"], size=(n_samples, 1))
        data = np.concatenate([data, cat], axis=1)
    return data


# Функция для оценки точности восстановления структуры
# known_edges: список кортежей (i, j) известных связей
# found_edges: список кортежей (i, j) найденных связей
# n_vars: число переменных


def evaluate_edges(known_edges, found_edges, n_vars):
    y_true = np.zeros((n_vars, n_vars), dtype=int)
    y_pred = np.zeros((n_vars, n_vars), dtype=int)
    for i, j in known_edges:
        y_true[i, j] = 1
    for i, j in found_edges:
        if i < n_vars and j < n_vars:
            y_pred[i, j] = 1
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()
    try:
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
    except Exception:
        precision, recall, f1 = 0.0, 0.0, 0.0
    return precision, recall, f1


# Сценарии бенчмарка
scenarios = [
    {
        "name": "Малый (100x5)",
        "n_samples": 100,
        "n_vars": 5,
        "noise": 0.1,
        "nan_ratio": 0.0,
        "categorical": False,
    },
    {
        "name": "Средний (1000x10)",
        "n_samples": 1000,
        "n_vars": 10,
        "noise": 0.1,
        "nan_ratio": 0.0,
        "categorical": False,
    },
    {
        "name": "Большой (10000x20)",
        "n_samples": 10000,
        "n_vars": 20,
        "noise": 0.1,
        "nan_ratio": 0.0,
        "categorical": False,
    },
    {
        "name": "Категориальные",
        "n_samples": 1000,
        "n_vars": 5,
        "noise": 0.1,
        "nan_ratio": 0.0,
        "categorical": True,
    },
    {
        "name": "С пропусками",
        "n_samples": 1000,
        "n_vars": 5,
        "noise": 0.1,
        "nan_ratio": 0.2,
        "categorical": False,
    },
    {
        "name": "С шумом",
        "n_samples": 1000,
        "n_vars": 5,
        "noise": 1.0,
        "nan_ratio": 0.0,
        "categorical": False,
    },
]

for scenario in scenarios:
    print(f"\nСценарий: {scenario['name']}")
    data_args = {k: v for k, v in scenario.items() if k != "name"}
    data = generate_data(**data_args)
    causal = CausalNova()
    start = time.time()
    try:
        causal.fit(data)
        elapsed = time.time() - start
        # Определяем число исходных признаков (до one-hot)
        n_vars = scenario["n_vars"]
        known_edges = [(0, 4)] if n_vars > 4 else []
        found_edges = [
            (i, j) for i, j in causal.graph.edges if i < n_vars and j < n_vars
        ]
        precision, recall, f1 = evaluate_edges(known_edges, found_edges, n_vars)
        print(f"Время работы: {elapsed:.3f} сек")
        print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}")
        print(f"Найденные связи: {found_edges}")
    except Exception as e:
        print(f"Ошибка: {e}")
