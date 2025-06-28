import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.stats import entropy
import logging
import pandas as pd

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class CausalNova:
    def __init__(self, tau=0.1, n_bootstraps=100):
        self.graph = nx.DiGraph()
        self.tau = tau
        self.n_bootstraps = n_bootstraps
        self.data = None
        logging.info("CausalNova инициализирована")

    def _validate_data(self, data):
        if not isinstance(data, np.ndarray):
            raise ValueError("Входные данные должны быть numpy.ndarray")
        if data.ndim != 2:
            raise ValueError("Входные данные должны быть двумерным массивом")
        if np.any(np.isinf(data)):
            raise ValueError("В данных присутствуют бесконечные значения")
        if data.shape[0] < 2 or data.shape[1] < 2:
            raise ValueError("Данные должны содержать минимум 2 строки и 2 столбца")
        logging.info("Входные данные прошли валидацию")

    def _preprocess(self, data):
        # Если данные не float, пробуем преобразовать через pandas
        if not np.issubdtype(data.dtype, np.number):
            df = pd.DataFrame(data)
            df = pd.get_dummies(df, dummy_na=True)
            logging.info(
                f"Категориальные признаки преобразованы: {df.shape[1]} столбцов"
            )
            return df.values.astype(float)
        return data

    def fit(self, data):
        # Преобразование категориальных данных
        data = self._preprocess(data)
        self._validate_data(data)
        self.data = data
        n, m = data.shape
        logging.info(f"Начало построения графа: {m} переменных")
        for i in range(m):
            for j in range(m):
                if i != j:
                    dep, pval = self._is_dependent(data[:, i], data[:, j])
                    if dep and pval <= 0.05:
                        self._add_edge(i, j)
        # Фильтрация слабых связей
        weak_edges = [
            (i, j)
            for i, j in self.graph.edges
            if self.graph.edges[i, j]["weight"] <= 0.2
        ]
        for edge in weak_edges:
            self.graph.remove_edge(*edge)
            logging.info(f"Удалено слабое ребро {edge[0]}->{edge[1]} (C_ij <= 0.2)")
        logging.info(f"Граф построен: {self.graph.number_of_edges()} ребер")

    def _hist_quantile_bins(self, arr, bins=10):
        arr = arr[~np.isnan(arr) & ~np.isinf(arr)]
        if len(arr) == 0:
            return np.array([1.0]), np.array([0.0])
        quantiles = np.linspace(0, 1, bins + 1)
        bin_edges = np.unique(np.quantile(arr, quantiles))
        if len(bin_edges) < 2:
            bin_edges = np.array([np.min(arr), np.max(arr) + 1e-6])
        hist, _ = np.histogram(arr, bins=bin_edges)
        return hist, bin_edges

    def _is_dependent(self, x, y):
        mask = ~np.isnan(x) & ~np.isnan(y) & ~np.isinf(x) & ~np.isinf(y)
        if np.sum(mask) < 10:
            return False, 1.0
        corr = np.abs(np.corrcoef(x[mask], y[mask])[0, 1])
        pval = 1 - corr
        logging.debug(f"Корреляция: {corr}, p-value: {pval}")
        return corr > 0.3, pval

    def _add_edge(self, i, j):
        try:
            delta_h = self._entropy_change(i, j)
            prob = np.exp(-delta_h / self.tau)
            rand_val = np.random.random()
            logging.debug(
                f"Проба направления {i}->{j}: ΔH={delta_h:.4f}, P={prob:.4f}, rand={rand_val:.4f}"
            )
            if rand_val < prob:
                weight = self._causal_strength(i, j)
                self.graph.add_edge(i, j, weight=weight)
                logging.info(f"Добавлено ребро {i}->{j} с весом {weight:.4f}")
        except Exception as e:
            logging.warning(f"Ошибка при добавлении ребра {i}->{j}: {e}")

    def _entropy_change(self, i, j):
        x = self.data[:, i]
        y = self.data[:, j]
        mask = ~np.isnan(x) & ~np.isnan(y) & ~np.isinf(x) & ~np.isinf(y)
        y_masked = y[mask]
        x_masked = x[mask]
        if len(y_masked) == 0:
            return 0.0
        h_before, _ = self._hist_quantile_bins(y_masked, bins=10)
        h_before = entropy(h_before)
        mask_high = x_masked > np.median(x_masked)
        if np.sum(mask_high) == 0:
            return 0.0
        h_after, _ = self._hist_quantile_bins(y_masked[mask_high], bins=10)
        h_after = entropy(h_after)
        delta = abs(h_after - h_before)
        logging.debug(f"ΔH для {i}->{j}: {delta:.4f}")
        return delta

    def _causal_strength(self, i, j):
        x = self.data[:, i]
        y = self.data[:, j]
        mask = ~np.isnan(x) & ~np.isnan(y) & ~np.isinf(x) & ~np.isinf(y)
        x_masked = x[mask]
        y_masked = y[mask]
        if len(x_masked) == 0 or len(y_masked) == 0:
            return 0.0
        cov = np.cov(x_masked, y_masked)[0, 1]
        sigma_i = np.std(x_masked)
        sigma_j = np.std(y_masked)
        h_cond = self._conditional_entropy(i, j)
        h_j, _ = self._hist_quantile_bins(y_masked, bins=10)
        h_j = entropy(h_j)
        strength = (cov / (sigma_i * sigma_j + 1e-10)) * np.exp(
            -(abs(h_cond - h_j) / self.tau)
        )
        logging.debug(f"Сила {i}->{j}: {strength:.4f}")
        return strength

    def _conditional_entropy(self, i, j):
        x = self.data[:, i]
        y = self.data[:, j]
        mask = ~np.isnan(x) & ~np.isnan(y) & ~np.isinf(x) & ~np.isinf(y)
        x_masked = x[mask]
        y_masked = y[mask]
        if len(x_masked) == 0 or len(y_masked) == 0:
            return 0.0
        # 2D квантили для совместной энтропии
        try:
            bins = 10
            x_edges = np.unique(np.quantile(x_masked, np.linspace(0, 1, bins + 1)))
            y_edges = np.unique(np.quantile(y_masked, np.linspace(0, 1, bins + 1)))
            if len(x_edges) < 2:
                x_edges = np.array([np.min(x_masked), np.max(x_masked) + 1e-6])
            if len(y_edges) < 2:
                y_edges = np.array([np.min(y_masked), np.max(y_masked) + 1e-6])
            h = entropy(
                np.histogram2d(x_masked, y_masked, bins=[x_edges, y_edges])[0].ravel()
            )
        except Exception as e:
            logging.warning(f"Ошибка при расчете условной энтропии: {e}")
            h = 0.0
        logging.debug(f"Условная энтропия {i}|{j}: {h:.4f}")
        return h

    def stability_test(self):
        stable_edges = {}
        for b in range(self.n_bootstraps):
            idx = np.random.choice(len(self.data), len(self.data), replace=True)
            sub_data = self.data[idx]
            sub_graph = nx.DiGraph()
            for i, j in self.graph.edges:
                try:
                    if self._is_dependent(sub_data[:, i], sub_data[:, j])[0]:
                        sub_graph.add_edge(i, j)
                except Exception as e:
                    logging.warning(f"Ошибка bootstrap {i}->{j}: {e}")
            for edge in self.graph.edges:
                stable_edges[edge] = stable_edges.get(edge, 0) + (
                    1 if edge in sub_graph.edges else 0
                )
            if b % 10 == 0:
                logging.info(f"Bootstrap {b}/{self.n_bootstraps}")
        return {edge: count / self.n_bootstraps for edge, count in stable_edges.items()}

    def explain(self):
        if self.graph.number_of_edges() == 0:
            logging.info("Граф пуст, объяснений нет")
            return "Нет причинно-следственных связей"
        explanation = []
        stability = self.stability_test()
        for i, j in self.graph.edges:
            weight = self.graph.edges[i, j]["weight"]
            stab = stability.get((i, j), 0)
            explanation.append(
                f"Связь {i} → {j}: сила {weight:.2f}, устойчивость {stab:.2f}"
            )
        logging.info("Генерация объяснения завершена")
        return "\n".join(explanation)

    def visualize(self):
        if self.graph.number_of_edges() == 0:
            logging.info("Граф пуст, визуализация невозможна")
            print("Нет причинно-следственных связей для визуализации")
            return
        pos = nx.spring_layout(self.graph)
        nx.draw(
            self.graph, pos, with_labels=True, node_color="lightgreen", node_size=500
        )
        edge_labels = nx.get_edge_attributes(self.graph, "weight")
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels)
        plt.title("Граф причинно-следственных связей")
        plt.show()


# Тест
if __name__ == "__main__":
    np.random.seed(42)
    data = np.random.rand(1000, 5)  # Пример данных
    data[:, 4] = 2 * data[:, 0] + np.random.rand(1000) * 0.1  # X0 → X4
    causal = CausalNova()
    causal.fit(data)
    print(causal.explain())
    causal.visualize()
