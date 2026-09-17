"""
Лабораторная работа zdop1_PCA.
Метод главных компонент (Principal Component Analysis, PCA).
Датасет: UCI Student Performance, student-mat.csv.

Файл должен находиться в одной папке с student-mat.csv.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score

# Общие настройки
DATA_FILE = Path("C:/Users/user/Downloads/student+performance/student/student-mat.csv")
OUT_DIR = Path("pca_outputs")
OUT_DIR.mkdir(exist_ok=True)
RANDOM_STATE = 42


# ============================================================
# ЗАДАЧА 1. Подготовка и краткое описание данных
# ============================================================
# student-mat.csv использует разделитель ';'.
df = pd.read_csv(DATA_FILE, sep=";")

print("ЗАДАЧА 1")
print("Размер исходного набора данных:", df.shape)
print("Количество пропусков:", int(df.isna().sum().sum()))
print("Первые 5 строк:")
print(df.head())

# G3 — итоговая оценка (переменная отклика). По требованию задания
# она удаляется из набора признаков перед выполнением PCA.
X_raw = df.drop(columns=["G3"])

# Для последующего сравнения кластеров с задачей классификации создаём
# пятиуровневые классы успеваемости из G3. Эти классы НЕ подаются в PCA.
# Шкала из сопровождающей публикации к UCI Student Performance:
# I: 16-20; II: 14-15; III: 12-13; IV: 10-11; V: 0-9.
y_class = pd.cut(
    df["G3"],
    bins=[-0.1, 9, 11, 13, 15, 20],
    labels=["V (0-9)", "IV (10-11)", "III (12-13)", "II (14-15)", "I (16-20)"],
    include_lowest=True,
)

# PCA работает с числовыми признаками. Категориальные признаки кодируем
# методом one-hot, удаляя одну фиктивную категорию в каждом признаке.
X = pd.get_dummies(X_raw, drop_first=True, dtype=float)

# Перед PCA стандартизируем признаки: среднее 0, стандартное отклонение 1.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nПосле удаления G3:", X_raw.shape)
print("После one-hot кодирования:", X.shape)
print("Распределение пяти классов G3:")
print(y_class.value_counts().sort_index())


# ============================================================
# ЗАДАЧА 2. PCA и кластеризация
# ============================================================
# Берём 3 главные компоненты. Это позволяет выполнить кластеризацию
# и затем визуализировать результат в задаче 3.
pca3 = PCA(n_components=3)
X_pca3 = pca3.fit_transform(X_scaled)

# Число кластеров k=5 выбрано для сопоставления с пятиуровневой
# классификацией успеваемости из сопровождающей публикации.
kmeans3 = KMeans(n_clusters=5, random_state=RANDOM_STATE, n_init=20)
clusters3 = kmeans3.fit_predict(X_pca3)

sil3 = silhouette_score(X_pca3, clusters3)
ari3 = adjusted_rand_score(y_class.astype(str), clusters3)

print("\nЗАДАЧА 2")
print(
    "Доли объяснённой дисперсии PC1-PC3:", np.round(pca3.explained_variance_ratio_, 4)
)
print(
    "Суммарная объяснённая дисперсия:", round(pca3.explained_variance_ratio_.sum(), 4)
)
print("Размеры кластеров:")
print(pd.Series(clusters3).value_counts().sort_index())
print("Silhouette score:", round(sil3, 4))
print("Adjusted Rand Index (сравнение с 5 классами G3):", round(ari3, 4))

# Таблица соответствия кластеров и пяти классов итоговой оценки.
comparison = pd.crosstab(
    pd.Series(clusters3, name="Кластер"),
    pd.Series(y_class.astype(str), name="Класс G3"),
)
print("\nТаблица соответствия кластеров и классов:")
print(comparison)
comparison.to_csv(OUT_DIR / "clusters_vs_classes.csv", encoding="utf-8-sig")

# Профили кластеров по нескольким исходным числовым признакам.
profile_columns = [
    "G1",
    "G2",
    "G3",
    "failures",
    "studytime",
    "absences",
    "goout",
    "Dalc",
    "Walc",
]
profiles = df[profile_columns].copy()
profiles["cluster"] = clusters3
cluster_profiles = profiles.groupby("cluster").mean().round(2)
print("\nСредние значения признаков по кластерам:")
print(cluster_profiles)
cluster_profiles.to_csv(OUT_DIR / "cluster_profiles.csv", encoding="utf-8-sig")


# ============================================================
# ЗАДАЧА 3. Визуализация для m <= 3 и сравнение с классификацией
# ============================================================
# --- m = 2 ---
pca2 = PCA(n_components=2)
X_pca2 = pca2.fit_transform(X_scaled)
kmeans2 = KMeans(n_clusters=5, random_state=RANDOM_STATE, n_init=20)
clusters2 = kmeans2.fit_predict(X_pca2)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].scatter(X_pca2[:, 0], X_pca2[:, 1], c=clusters2, cmap="tab10", s=28, alpha=0.8)
axes[0].set_title("PCA, m=2: цвета = номера кластеров KMeans")
axes[0].set_xlabel("PC1")
axes[0].set_ylabel("PC2")
axes[0].grid(alpha=0.2)

class_codes = y_class.cat.codes
scatter = axes[1].scatter(
    X_pca2[:, 0], X_pca2[:, 1], c=class_codes, cmap="tab10", s=28, alpha=0.8
)
axes[1].set_title("PCA, m=2: цвета = классы итоговой оценки G3")
axes[1].set_xlabel("PC1")
axes[1].set_ylabel("PC2")
axes[1].grid(alpha=0.2)

handles, _ = scatter.legend_elements(num=5)
axes[1].legend(
    handles, [str(x) for x in y_class.cat.categories], title="Класс G3", fontsize=8
)
plt.tight_layout()
plt.savefig(OUT_DIR / "task3_pca_2d.png", dpi=180, bbox_inches="tight")
plt.show()

# --- m = 3 ---
fig = plt.figure(figsize=(13, 5))
ax1 = fig.add_subplot(121, projection="3d")
ax1.scatter(
    X_pca3[:, 0], X_pca3[:, 1], X_pca3[:, 2], c=clusters3, cmap="tab10", s=24, alpha=0.8
)
ax1.set_title("PCA, m=3: кластеры KMeans")
ax1.set_xlabel("PC1")
ax1.set_ylabel("PC2")
ax1.set_zlabel("PC3")

ax2 = fig.add_subplot(122, projection="3d")
sc = ax2.scatter(
    X_pca3[:, 0],
    X_pca3[:, 1],
    X_pca3[:, 2],
    c=class_codes,
    cmap="tab10",
    s=24,
    alpha=0.8,
)
ax2.set_title("PCA, m=3: классы итоговой оценки G3")
ax2.set_xlabel("PC1")
ax2.set_ylabel("PC2")
ax2.set_zlabel("PC3")
handles, _ = sc.legend_elements(num=5)
ax2.legend(
    handles, [str(x) for x in y_class.cat.categories], title="Класс G3", fontsize=7
)
plt.tight_layout()
plt.savefig(OUT_DIR / "task3_pca_3d.png", dpi=180, bbox_inches="tight")
plt.show()

print("\nЗАДАЧА 3")
print(
    "m=2: explained variance =",
    round(pca2.explained_variance_ratio_.sum(), 4),
    "; silhouette =",
    round(silhouette_score(X_pca2, clusters2), 4),
    "; ARI =",
    round(adjusted_rand_score(y_class.astype(str), clusters2), 4),
)
print(
    "m=3: explained variance =",
    round(pca3.explained_variance_ratio_.sum(), 4),
    "; silhouette =",
    round(sil3, 4),
    "; ARI =",
    round(ari3, 4),
)


# ============================================================
# ЗАДАЧА 4. Исследование влияния параметров PCA
# ============================================================
# Варьируем n_components и параметр whiten. Для каждого варианта
# выполняем KMeans(k=5) и считаем объяснённую дисперсию, silhouette и ARI.
results = []
for n_components in [2, 3, 5, 10, 15, 20]:
    for whiten in [False, True]:
        pca = PCA(n_components=n_components, whiten=whiten)
        X_current = pca.fit_transform(X_scaled)
        km = KMeans(n_clusters=5, random_state=RANDOM_STATE, n_init=20)
        current_clusters = km.fit_predict(X_current)

        results.append(
            {
                "n_components": n_components,
                "whiten": whiten,
                "explained_variance": pca.explained_variance_ratio_.sum(),
                "silhouette": silhouette_score(X_current, current_clusters),
                "ARI": adjusted_rand_score(y_class.astype(str), current_clusters),
            }
        )

results_df = pd.DataFrame(results)
print("\nЗАДАЧА 4")
print(results_df.round(4).to_string(index=False))
results_df.to_csv(
    OUT_DIR / "pca_parameter_results.csv", index=False, encoding="utf-8-sig"
)

# Отдельно исследуем, сколько компонент нужно для заданной доли дисперсии.
pca_full = PCA()
pca_full.fit(X_scaled)
cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)

for threshold in [0.80, 0.90, 0.95]:
    required = int(np.searchsorted(cumulative_variance, threshold) + 1)
    print(
        f"Для сохранения не менее {threshold:.0%} дисперсии требуется компонент: {required}"
    )

plt.figure(figsize=(8, 5))
plt.plot(
    range(1, len(cumulative_variance) + 1),
    cumulative_variance,
    marker="o",
    markersize=3,
)
plt.axhline(0.80, linestyle="--", linewidth=1, label="80%")
plt.axhline(0.90, linestyle="--", linewidth=1, label="90%")
plt.axhline(0.95, linestyle="--", linewidth=1, label="95%")
plt.xlabel("Число главных компонент")
plt.ylabel("Накопленная объяснённая дисперсия")
plt.title("Накопленная объяснённая дисперсия PCA")
plt.grid(alpha=0.25)
plt.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "task4_cumulative_variance.png", dpi=180, bbox_inches="tight")
plt.show()

print("\nГотово. Результаты и изображения сохранены в папку:", OUT_DIR.resolve())
