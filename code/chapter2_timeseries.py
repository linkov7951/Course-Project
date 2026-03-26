import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy import stats
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ========== НАСТРОЙКА ПУТЕЙ ==========
data_path = 'D:/2 семестр/курсач/data/timeseries/nvidia_stock.csv'
results_path = 'D:/2 семестр/курсач/results/'
os.makedirs(results_path, exist_ok=True)

print("=" * 60)
print("ГЛАВА 2: АНАЛИЗ ВРЕМЕННЫХ РЯДОВ")
print("=" * 60)

# ========== ЗАГРУЗКА ДАННЫХ ==========
print("\n1. Загрузка данных...")
try:
    df = pd.read_csv(data_path, skiprows=2)
    print(f"   Данные загружены: {df.shape[0]} строк, {df.shape[1]} столбцов")
    
    df.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
    print(f"   Колонки после переименования: {list(df.columns)}")
    
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)
    
    print(f"\n   Диапазон дат: с {df.index.min()} по {df.index.max()}")
    print(f"   Количество торговых дней: {len(df)}")
    
    price_columns = ['Open', 'High', 'Low', 'Close']
    for col in price_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
    df = df.dropna()
    print(f"   После очистки: {len(df)} строк")
    
except FileNotFoundError:
    print(f"   ОШИБКА: Файл не найден по пути {data_path}")
    print("   Создаю синтетические данные для демонстрации...")
    
    np.random.seed(42)
    dates = pd.date_range(start='2016-01-01', end='2026-01-01', freq='B')
    n = len(dates)
    
    trend = 30 * np.exp(np.linspace(0, 3.5, n))
    seasonal = 10 * np.sin(np.linspace(0, 40*np.pi, n))
    noise = np.random.normal(0, 5, n)
    
    close_prices = trend + seasonal + noise
    open_prices = close_prices + np.random.normal(0, 2, n)
    high_prices = np.maximum(close_prices, open_prices) + np.abs(np.random.normal(0, 3, n))
    low_prices = np.minimum(close_prices, open_prices) - np.abs(np.random.normal(0, 3, n))
    volumes = np.random.lognormal(18, 0.5, n) * (1 + 0.5 * np.sin(np.linspace(0, 40*np.pi, n)))
    
    df = pd.DataFrame({
        'Date': dates,
        'Open': open_prices,
        'High': high_prices,
        'Low': low_prices,
        'Close': close_prices,
        'Volume': volumes
    })
    df.set_index('Date', inplace=True)
    print(f"   Создано {len(df)} синтетических записей")

print(f"\n   Первые 5 строк:")
print(df.head())

# ========== 1. ВИЗУАЛИЗАЦИЯ ВСЕХ КАНАЛОВ ==========
print("\n2. Визуализация многомерного ряда...")

fig, axes = plt.subplots(5, 1, figsize=(15, 14), sharex=True)

axes[0].plot(df.index, df['Close'], color='blue', linewidth=1)
axes[0].set_title('Цена закрытия (Close)')
axes[0].set_ylabel('Цена ($)')
axes[0].grid(True, alpha=0.3)

axes[1].plot(df.index, df['Open'], color='green', linewidth=1)
axes[1].set_title('Цена открытия (Open)')
axes[1].set_ylabel('Цена ($)')
axes[1].grid(True, alpha=0.3)

axes[2].plot(df.index, df['High'], color='red', linewidth=1)
axes[2].set_title('Максимальная цена (High)')
axes[2].set_ylabel('Цена ($)')
axes[2].grid(True, alpha=0.3)

axes[3].plot(df.index, df['Low'], color='orange', linewidth=1)
axes[3].set_title('Минимальная цена (Low)')
axes[3].set_ylabel('Цена ($)')
axes[3].grid(True, alpha=0.3)

axes[4].plot(df.index, df['Volume'] / 1e6, color='purple', linewidth=1)
axes[4].set_title('Объем торгов (Volume, млн)')
axes[4].set_ylabel('Объем (млн)')
axes[4].set_xlabel('Год')
axes[4].grid(True, alpha=0.3)

plt.suptitle('Рисунок 2.1 - Многомерный временной ряд акций NVIDIA (2016-2026)', fontsize=14, y=0.995)
plt.tight_layout()
plt.savefig(os.path.join(results_path, 'fig2_1_multivariate_ts.png'), dpi=300, bbox_inches='tight')
plt.show()

# ========== 2. СТАТИСТИЧЕСКИЙ АНАЛИЗ ==========
print("\n3. Статистический анализ...")

stats_df = df.describe().T
stats_df['range'] = stats_df['max'] - stats_df['min']
stats_df['cv'] = (stats_df['std'] / stats_df['mean'] * 100).round(2)

print("\nТаблица 2.5 - Статистические характеристики каналов")
print(stats_df[['mean', '50%', 'std', 'min', 'max', 'range', 'cv']].round(2))

date_diff = df.index.to_series().diff().dt.days.dropna()
print(f"\n   Средний интервал между наблюдениями: {date_diff.mean():.2f} дней")
print(f"   Медианный интервал: {date_diff.median():.2f} дней")
print(f"   Максимальный интервал: {date_diff.max():.2f} дней")

# ========== 3. ПРОПУСКИ И ВЫБРОСЫ ==========
print("\n4. Анализ пропусков и выбросов...")

missing_values = df.isnull().sum()
print("\nТаблица 2.7 - Пропуски в данных")
for col in df.columns:
    print(f"   {col}: {missing_values[col]} пропусков")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].boxplot(df['Close'])
axes[0].set_title('Цена закрытия')
axes[0].set_ylabel('Цена ($)')
axes[0].grid(True, alpha=0.3)

axes[1].boxplot(df['Volume'] / 1e6)
axes[1].set_title('Объем торгов')
axes[1].set_ylabel('Объем (млн)')
axes[1].grid(True, alpha=0.3)

plt.suptitle('Рисунок 2.2 - Диаграммы размаха', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(results_path, 'fig2_2_boxplots.png'), dpi=300, bbox_inches='tight')
plt.show()

# ========== 4. РИСУНОК 2.2.1 - АНОМАЛИИ ОБЪЕМА ==========
print("\n5. Создание рисунка 2.2.1 - Временной ряд с выделением аномальных значений объема...")

anomaly_threshold = 3e9
anomalies = df[df['Volume'] > anomaly_threshold]

plt.figure(figsize=(15, 6))

normal_volume = df[df['Volume'] <= anomaly_threshold]
plt.plot(normal_volume.index, normal_volume['Volume'] / 1e6, 
         color='blue', linewidth=1, label='Объем торгов')

if len(anomalies) > 0:
    plt.scatter(anomalies.index, anomalies['Volume'] / 1e6, 
                color='red', s=50, zorder=5, label=f'Аномальные значения (>3 млрд)')

plt.axhline(y=anomaly_threshold/1e6, color='red', linestyle='--', alpha=0.5, 
            label=f'Порог аномалии ({anomaly_threshold/1e6:.0f} млн)')

plt.title('Рисунок 2.2.1 - Временной ряд с выделением аномальных значений объема', fontsize=14)
plt.xlabel('Год')
plt.ylabel('Объем (млн)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(results_path, 'fig2_2.1_volume_anomalies.png'), dpi=300, bbox_inches='tight')
plt.show()
print(f"   Рисунок сохранен: fig2_2.1_volume_anomalies.png")

# ========== 5. ДИАПАЗОНЫ ЗНАЧЕНИЙ ==========
print("\n6. Анализ диапазонов значений...")

print("\nТаблица 2.8 - Диапазоны значений каналов")
for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
    if col == 'Volume':
        print(f"   {col}: min={df[col].min()/1e6:.2f}M, max={df[col].max()/1e6:.2f}M, range={(df[col].max()-df[col].min())/1e6:.2f}M")
    else:
        print(f"   {col}: min={df[col].min():.2f}, max={df[col].max():.2f}, range={df[col].max()-df[col].min():.2f}")

scaler = StandardScaler()
df_scaled = df.copy()
df_scaled[['Open', 'High', 'Low', 'Close', 'Volume']] = scaler.fit_transform(df[['Open', 'High', 'Low', 'Close', 'Volume']])

plt.figure(figsize=(15, 8))
for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
    plt.plot(df_scaled.index, df_scaled[col], label=col, linewidth=0.8)

plt.title('Рисунок 2.3 - Все каналы временного ряда после стандартизации', fontsize=14)
plt.xlabel('Год')
plt.ylabel('Нормированное значение')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(results_path, 'fig2_3_scaled_channels.png'), dpi=300, bbox_inches='tight')
plt.show()

# ========== 6. КОРРЕЛЯЦИОННЫЙ АНАЛИЗ ==========
print("\n7. Корреляционный анализ...")

corr_matrix = df[['Open', 'High', 'Low', 'Close', 'Volume']].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
            square=True, linewidths=1, fmt='.3f',
            cbar_kws={"shrink": 0.8})
plt.title('Рисунок 2.4 - Матрица корреляций каналов временного ряда', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(results_path, 'fig2_4_correlation_heatmap.png'), dpi=300, bbox_inches='tight')
plt.show()

print("\nТаблица 2.9 - Матрица корреляций Пирсона")
print(corr_matrix.round(3))

# ========== 7. ДЕКОМПОЗИЦИЯ И АНАЛИЗ ШУМОВ ==========
print("\n8. Декомпозиция и анализ шумов...")

close_series = df['Close'].dropna()
period = min(60, len(close_series) // 10)

try:
    decomposition = seasonal_decompose(close_series, model='additive', period=period)
    
    fig, axes = plt.subplots(4, 1, figsize=(15, 12))
    
    axes[0].plot(close_series.index, close_series, color='black', linewidth=1)
    axes[0].set_title('Исходный ряд')
    axes[0].set_ylabel('Цена ($)')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(decomposition.trend.index, decomposition.trend, color='blue', linewidth=1)
    axes[1].set_title('Тренд')
    axes[1].set_ylabel('Цена ($)')
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(decomposition.seasonal.index, decomposition.seasonal, color='green', linewidth=1)
    axes[2].set_title('Сезонность')
    axes[2].set_ylabel('Амплитуда')
    axes[2].grid(True, alpha=0.3)
    
    axes[3].plot(decomposition.resid.index, decomposition.resid, color='red', linewidth=1)
    axes[3].set_title('Остатки (шум)')
    axes[3].set_ylabel('Остатки')
    axes[3].set_xlabel('Год')
    axes[3].grid(True, alpha=0.3)
    
    plt.suptitle('Рисунок 2.5 - Декомпозиция временного ряда цены закрытия', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, 'fig2_5_decomposition.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    signal = decomposition.trend + decomposition.seasonal
    signal_var = np.nanvar(signal)
    noise_var = np.nanvar(decomposition.resid)
    snr_db = 10 * np.log10(signal_var / noise_var)
    
    print(f"\n   Дисперсия сигнала: {signal_var:.2f}")
    print(f"   Дисперсия шума: {noise_var:.2f}")
    print(f"   Отношение сигнал/шум (SNR): {snr_db:.2f} дБ")
    
    plt.figure(figsize=(12, 6))
    plt.hist(decomposition.resid.dropna(), bins=50, edgecolor='black', alpha=0.7, color='red')
    plt.title('Рисунок 2.6 - Гистограмма распределения остатков (шума)', fontsize=14)
    plt.xlabel('Значение остатков')
    plt.ylabel('Частота')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(results_path, 'fig2_6_noise_histogram.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    resid_clean = decomposition.resid.dropna()
    if len(resid_clean) > 0:
        ks_statistic, p_value = stats.kstest(resid_clean, 'norm', args=(resid_clean.mean(), resid_clean.std()))
        print(f"   KS-тест на нормальность: статистика={ks_statistic:.4f}, p-value={p_value:.4f}")
        
        if p_value > 0.05:
            print("   Распределение близко к нормальному (p > 0.05)")
        else:
            print("   Распределение отличается от нормального (p < 0.05)")

except Exception as e:
    print(f"   Ошибка при декомпозиции: {e}")

# ========== 8. АНАЛИЗ ВОЛАТИЛЬНОСТИ ==========
print("\n9. Анализ волатильности...")

df['returns'] = df['Close'].pct_change() * 100
df['volatility'] = df['returns'].rolling(window=30).std()

plt.figure(figsize=(15, 6))
plt.plot(df.index, df['volatility'], color='purple', linewidth=1)
plt.title('30-дневная волатильность (скользящее стандартное отклонение доходности)')
plt.xlabel('Год')
plt.ylabel('Волатильность (%)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(results_path, 'fig2_7_volatility.png'), dpi=300, bbox_inches='tight')
plt.show()

# ========== 9. РИСУНОК 2.8 - СРАВНЕНИЕ ДО И ПОСЛЕ 2020 ==========
print("\n10. Создание рисунка 2.8 - Сравнение распределений цены до и после 2020 года...")

df_before = df[df.index < '2020-01-01']
df_after = df[df.index >= '2020-01-01']

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(df_before['Close'], bins=50, alpha=0.7, color='blue', edgecolor='black')
axes[0].set_title(f'Распределение цены до 2020 года\nсреднее: {df_before["Close"].mean():.2f}$', fontsize=12)
axes[0].set_xlabel('Цена закрытия ($)')
axes[0].set_ylabel('Частота')
axes[0].grid(True, alpha=0.3)
axes[0].axvline(df_before['Close'].mean(), color='red', linestyle='--', linewidth=2)

axes[1].hist(df_after['Close'], bins=50, alpha=0.7, color='green', edgecolor='black')
axes[1].set_title(f'Распределение цены после 2020 года\nсреднее: {df_after["Close"].mean():.2f}$', fontsize=12)
axes[1].set_xlabel('Цена закрытия ($)')
axes[1].set_ylabel('Частота')
axes[1].grid(True, alpha=0.3)
axes[1].axvline(df_after['Close'].mean(), color='red', linestyle='--', linewidth=2)

plt.suptitle('Рисунок 2.8 - Сравнение распределений цены до и после 2020 года', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(results_path, 'fig2_8_price_comparison.png'), dpi=300, bbox_inches='tight')
plt.show()
print(f"   Рисунок сохранен: fig2_8_price_comparison.png")

print("\n" + "=" * 60)
print("АНАЛИЗ ВРЕМЕННЫХ РЯДОВ ЗАВЕРШЕН")
print(f"Все рисунки сохранены в: {results_path}")
print("=" * 60)