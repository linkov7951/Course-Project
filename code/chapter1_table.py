import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

# ========== НАСТРОЙКА ПУТЕЙ ==========
data_path = 'D:/2 семестр/курсач/data/table/creditcard.csv'
results_path = 'D:/2 семестр/курсач/results/'
os.makedirs(results_path, exist_ok=True)

print("=" * 60)
print("ГЛАВА 1: АНАЛИЗ ТАБЛИЧНЫХ ДАННЫХ")
print("=" * 60)

# ========== ЗАГРУЗКА ДАННЫХ ==========
print("\n1. Загрузка данных...")
try:
    df = pd.read_csv(data_path)
    print(f"   Данные загружены: {df.shape[0]} строк, {df.shape[1]} столбцов")
    print(f"   Первые 5 строк:")
    print(df.head())
    
    # Информация о датасете
    print(f"\n   Мошеннических транзакций: {df['Class'].sum()} из {len(df)}")
    print(f"   Доля мошенничества: {df['Class'].mean()*100:.3f}%")
    
except FileNotFoundError:
    print(f"   ОШИБКА: Файл не найден по пути {data_path}")
    exit()

# ========== ЧИСЛОВЫЕ ПРИЗНАКИ ==========
v_features = [f'V{i}' for i in range(1, 29)]
numeric_features = ['Time', 'Amount'] + v_features

# ========== 1. ГИСТОГРАММЫ РАСПРЕДЕЛЕНИЙ ==========
print("\n2. Построение гистограмм распределений...")

key_features = ['Time', 'Amount'] + [f'V{i}' for i in range(1, 6)]

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

for i, feature in enumerate(key_features):
    df[feature].hist(ax=axes[i], bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    axes[i].set_title(f'Распределение признака {feature}')
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel('Частота')
    axes[i].grid(True, alpha=0.3)

for i in range(len(key_features), len(axes)):
    axes[i].axis('off')

plt.suptitle('Рисунок 1.1 - Гистограммы распределения ключевых признаков', y=0.98, fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(results_path, 'fig1_1_distributions.png'), dpi=300, bbox_inches='tight')
plt.show()

# ========== 2. SEABORN ВИЗУАЛИЗАЦИЯ ==========
print("\n3. Построение pairplot...")
sample_df = df.sample(min(5000, len(df)))
features_to_plot = ['Time', 'Amount', 'V1', 'V2', 'V3', 'Class']

if all(f in df.columns for f in features_to_plot):
    sns.pairplot(sample_df[features_to_plot], 
                 hue='Class', 
                 diag_kind='hist',
                 palette={0: 'blue', 1: 'red'},
                 height=2)
    plt.suptitle('Рисунок 1.2 - Взаимосвязи признаков с окраской по классу', y=1.02, fontsize=14)
    plt.savefig(os.path.join(results_path, 'fig1_2_pairplot.png'), dpi=300, bbox_inches='tight')
    plt.show()

# ========== 3. ИНТЕРАКТИВНЫЙ ГРАФИК ==========
print("\n4. Создание интерактивного графика...")
fig = px.scatter(sample_df.sample(min(1000, len(sample_df))), 
                 x='Time', 
                 y='Amount', 
                 color='Class',
                 hover_data=['V1', 'V2'],
                 title='Рисунок 1.3 - Интерактивный график: сумма транзакции от времени',
                 labels={'Time': 'Время (сек)', 'Amount': 'Сумма транзакции'})
fig.write_html(os.path.join(results_path, 'fig1_3_interactive.html'))
print(f"   Интерактивный график сохранен: {os.path.join(results_path, 'fig1_3_interactive.html')}")

# ========== 4. АНАЛИЗ ПРОПУСКОВ ==========
print("\n5. Анализ пропусков...")
missing_data = pd.DataFrame({
    'Признак': df.columns,
    'Количество пропусков': df.isnull().sum().values,
    'Доля пропусков (%)': (df.isnull().sum() / len(df) * 100).values
})
print("\nТаблица 1.6 - Анализ пропущенных значений")
print(missing_data.to_string(index=False))

# ========== 5. КОРРЕЛЯЦИОННЫЙ АНАЛИЗ ==========
print("\n6. Корреляционный анализ...")
corr_with_target = df.corrwith(df['Class']).sort_values(ascending=False).to_frame()
corr_with_target.columns = ['Correlation with Class']

plt.figure(figsize=(10, 12))
sns.heatmap(corr_with_target, annot=True, cmap='coolwarm', center=0,
            linewidths=1, fmt='.3f', cbar_kws={"shrink": 0.8})
plt.title('Рисунок 1.4 - Корреляция признаков с целевой переменной (Class)', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(results_path, 'fig1_4_correlation.png'), dpi=300, bbox_inches='tight')
plt.show()

print("\nТаблица 1.7 - Корреляция с целевой переменной")
print(corr_with_target.head(10))

# ========== 6. ДУБЛИКАТЫ ==========
print("\n7. Проверка дубликатов...")
duplicates_count = df.duplicated().sum()
print(f"   Количество дубликатов: {duplicates_count}")

if duplicates_count > 0:
    df_clean = df.drop_duplicates()
    print(f"   Размер после удаления: {df_clean.shape}")
else:
    df_clean = df.copy()
    print("   Дубликаты не обнаружены")

# ========== 7. АНАЛИЗ ВЫБРОСОВ ==========
print("\n8. Анализ выбросов...")

def count_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return len(outliers), lower_bound, upper_bound

outliers_info = []
for feature in ['Time', 'Amount']:
    count, lower, upper = count_outliers_iqr(df_clean, feature)
    outliers_info.append({
        'Признак': feature,
        'Количество выбросов': count,
        'Доля выбросов (%)': round((count / len(df_clean)) * 100, 2),
        'Нижняя граница': round(lower, 2),
        'Верхняя граница': round(upper, 2)
    })

outliers_df = pd.DataFrame(outliers_info)
print("\nТаблица 1.8 - Анализ выбросов по методу IQR")
print(outliers_df.to_string(index=False))

# Box plots
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].boxplot(df_clean['Amount'])
axes[0].set_title('Amount')
axes[0].set_ylabel('Сумма транзакции')
axes[0].grid(True, alpha=0.3)

axes[1].boxplot(df_clean['Time'])
axes[1].set_title('Time')
axes[1].set_ylabel('Время (сек)')
axes[1].grid(True, alpha=0.3)

plt.suptitle('Рисунок 1.5 - Диаграммы размаха для Amount и Time', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(results_path, 'fig1_5_boxplots.png'), dpi=300, bbox_inches='tight')
plt.show()

# ========== 8. ФИЛЬТРАЦИЯ ==========
print("\n9. Фильтрация данных...")

large_threshold = df_clean['Amount'].quantile(0.9)
large_transactions = df_clean[df_clean['Amount'] > large_threshold]
print(f"   Крупных транзакций (> {large_threshold:.2f}): {len(large_transactions)} ({len(large_transactions)/len(df_clean)*100:.2f}%)")

fraud_transactions = df_clean[df_clean['Class'] == 1]
print(f"   Мошеннических транзакций: {len(fraud_transactions)} ({len(fraud_transactions)/len(df_clean)*100:.3f}%)")

df_clean['Hour'] = (df_clean['Time'] / 3600) % 24
night_transactions = df_clean[(df_clean['Hour'] >= 0) & (df_clean['Hour'] <= 6)]
print(f"   Ночных транзакций (0-6 часов): {len(night_transactions)} ({len(night_transactions)/len(df_clean)*100:.2f}%)")

# ========== 9. ДОБАВЛЕНИЕ ШУМА ==========
print("\n10. Добавление шума...")

df_noisy = df_clean.copy()
amount_std = df_noisy['Amount'].std()
amount_noise = np.random.normal(0, 0.02 * amount_std, len(df_noisy))
df_noisy['Amount_noisy'] = df_noisy['Amount'] + amount_noise
df_noisy['Amount_noisy'] = df_noisy['Amount_noisy'].clip(lower=0)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].hist(df_noisy['Amount'], bins=50, alpha=0.7, color='blue', edgecolor='black')
axes[0].set_title('Распределение Amount (оригинал)')
axes[0].set_xlabel('Сумма')
axes[0].set_ylabel('Частота')

axes[1].hist(df_noisy['Amount_noisy'], bins=50, alpha=0.7, color='orange', edgecolor='black')
axes[1].set_title('Распределение Amount с шумом')
axes[1].set_xlabel('Сумма')
axes[1].set_ylabel('Частота')

plt.suptitle('Рисунок 1.6 - Сравнение распределений до и после добавления шума', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(results_path, 'fig1_6_noise_comparison.png'), dpi=300, bbox_inches='tight')
plt.show()

# ========== 10. ПРЕОБРАЗОВАНИЕ В КАТЕГОРИИ ==========
print("\n11. Преобразование в категориальные признаки...")

df_clean['Amount_category'] = pd.cut(df_clean['Amount'], 
                                      bins=[0, 10, 50, 100, 200, 500, 10000],
                                      labels=['micro', 'very_small', 'small', 'medium', 'large', 'very_large'])

def time_to_period(time_sec):
    hour = (time_sec / 3600) % 24
    if 6 <= hour < 12:
        return 'morning'
    elif 12 <= hour < 18:
        return 'afternoon'
    elif 18 <= hour < 24:
        return 'evening'
    else:
        return 'night'

df_clean['day_period'] = df_clean['Time'].apply(time_to_period)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

df_clean['Amount_category'].value_counts().sort_index().plot(kind='bar', ax=axes[0], color='skyblue', edgecolor='black')
axes[0].set_title('Распределение категорий сумм')
axes[0].set_xlabel('Категория')
axes[0].set_ylabel('Количество')
axes[0].tick_params(axis='x', rotation=45)

df_clean['day_period'].value_counts().plot(kind='bar', ax=axes[1], color='lightgreen', edgecolor='black')
axes[1].set_title('Распределение по периодам дня')
axes[1].set_xlabel('Период')
axes[1].set_ylabel('Количество')
axes[1].tick_params(axis='x', rotation=0)

plt.suptitle('Рисунок 1.7 - Распределение новых категориальных признаков', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(results_path, 'fig1_7_categorical_distributions.png'), dpi=300, bbox_inches='tight')
plt.show()

# ========== 11. СРАВНЕНИЕ МОШЕННИЧЕСКИХ И ОБЫЧНЫХ ТРАНЗАКЦИЙ ==========
print("\n12. Анализ по подгруппам...")

fraud = df_clean[df_clean['Class'] == 1]
normal = df_clean[df_clean['Class'] == 0]

print("\nТаблица 1.9 - Сравнение статистик для мошеннических и обычных транзакций")
print(f"{'Показатель':<30} {'Обычные':<15} {'Мошеннические':<15} {'Отношение':<10}")
print("-" * 70)
print(f"{'Количество':<30} {len(normal):<15} {len(fraud):<15} {len(fraud)/len(normal)*100:.3f}%")
print(f"{'Средняя сумма':<30} {normal['Amount'].mean():.2f} {fraud['Amount'].mean():.2f} {fraud['Amount'].mean()/normal['Amount'].mean():.2f}×")
print(f"{'Медианная сумма':<30} {normal['Amount'].median():.2f} {fraud['Amount'].median():.2f} {fraud['Amount'].median()/normal['Amount'].median():.2f}×")
print(f"{'Среднее время':<30} {normal['Time'].mean():.0f} {fraud['Time'].mean():.0f} {fraud['Time'].mean()/normal['Time'].mean():.2f}×")

# ========== 12. РИСУНОК 1.5.1 - ВЫБРОСЫ ==========
print("\n13. Создание рисунка 1.5.1 - Распределение Amount с выделением выбросов...")

Q1 = df_clean['Amount'].quantile(0.25)
Q3 = df_clean['Amount'].quantile(0.75)
IQR = Q3 - Q1
upper_bound = Q3 + 1.5 * IQR

normal_data = df_clean[df_clean['Amount'] <= upper_bound]['Amount']
outlier_data = df_clean[df_clean['Amount'] > upper_bound]['Amount']

plt.figure(figsize=(12, 6))
plt.hist(normal_data, bins=50, alpha=0.7, color='blue', edgecolor='black', 
         label=f'Нормальные значения (n={len(normal_data)})')
plt.hist(outlier_data, bins=50, alpha=0.7, color='red', edgecolor='black',
         label=f'Выбросы (n={len(outlier_data)})')
plt.title('Рисунок 1.5.1 - Распределение Amount с выделением выбросов', fontsize=14)
plt.xlabel('Сумма транзакции')
plt.ylabel('Частота')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(results_path, 'fig1_5.1_boxplots_with_outliers.png'), dpi=300, bbox_inches='tight')
plt.show()
print(f"   Рисунок сохранен: fig1_5.1_boxplots_with_outliers.png")

# ========== 13. ФИНАЛЬНОЕ СРАВНЕНИЕ ==========
print("\n14. Финальное сравнение распределений...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].hist(np.log1p(df['Amount']), bins=50, alpha=0.7, color='blue', edgecolor='black')
axes[0].set_title('Исходные данные: log(Amount)')
axes[0].set_xlabel('log(сумма)')
axes[0].set_ylabel('Частота')

axes[1].hist(np.log1p(df_noisy['Amount_noisy']), bins=50, alpha=0.7, color='green', edgecolor='black')
axes[1].set_title('После преобразований: log(Amount_noisy)')
axes[1].set_xlabel('log(сумма)')
axes[1].set_ylabel('Частота')

plt.suptitle('Рисунок 1.8 - Сравнение исходных и преобразованных данных', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(results_path, 'fig1_8_before_after.png'), dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "=" * 60)
print("АНАЛИЗ ТАБЛИЧНЫХ ДАННЫХ ЗАВЕРШЕН")
print(f"Все рисунки сохранены в: {results_path}")
print("=" * 60)

# ========== Рисунок 1.9 - Сравнение статистик мошеннических и обычных транзакций ==========
print("\nСоздание рисунка 1.9 - Сравнение статистик мошеннических и обычных транзакций...")

# Данные для визуализации
categories = ['Средняя сумма', 'Медианная сумма', 'Среднее время']
fraud_values = [122.21, 9.69, 80761]
normal_values = [88.29, 22.00, 94848]

# Нормализуем время для отображения на одном графике (делим на 1000)
fraud_time_norm = 80761 / 1000
normal_time_norm = 94848 / 1000

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# График 1: Средняя сумма
bars1 = axes[0].bar(['Обычные', 'Мошеннические'], [88.29, 122.21], 
                     color=['blue', 'red'], edgecolor='black', alpha=0.7)
axes[0].set_title('Средняя сумма транзакции', fontsize=12)
axes[0].set_ylabel('Сумма ($)')
axes[0].grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars1, [88.29, 122.21]):
    height = bar.get_height()
    axes[0].text(bar.get_x() + bar.get_width()/2., height + 1, f'{val:.2f}', 
                 ha='center', fontweight='bold')

# График 2: Медианная сумма
bars2 = axes[1].bar(['Обычные', 'Мошеннические'], [22.00, 9.69], 
                     color=['blue', 'red'], edgecolor='black', alpha=0.7)
axes[1].set_title('Медианная сумма транзакции', fontsize=12)
axes[1].set_ylabel('Сумма ($)')
axes[1].grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars2, [22.00, 9.69]):
    height = bar.get_height()
    axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.3, f'{val:.2f}', 
                 ha='center', fontweight='bold')

# График 3: Среднее время
bars3 = axes[2].bar(['Обычные', 'Мошеннические'], [94.85, 80.76], 
                     color=['blue', 'red'], edgecolor='black', alpha=0.7)
axes[2].set_title('Среднее время транзакции (тыс. сек)', fontsize=12)
axes[2].set_ylabel('Время (тыс. сек)')
axes[2].grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars3, [94.85, 80.76]):
    height = bar.get_height()
    axes[2].text(bar.get_x() + bar.get_width()/2., height + 1, f'{val:.2f}', 
                 ha='center', fontweight='bold')

plt.suptitle('Рисунок 1.9 - Сравнение статистик мошеннических и обычных транзакций', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(results_path, 'fig1_9_comparison.png'), dpi=300, bbox_inches='tight')
plt.show()
print(f"   Рисунок сохранен: fig1_9_comparison.png")