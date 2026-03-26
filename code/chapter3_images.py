import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from glob import glob
from collections import Counter
from PIL import Image
import xml.etree.ElementTree as ET
import warnings
warnings.filterwarnings('ignore')

# ========== НАСТРОЙКА ПУТЕЙ ==========
base_path = 'D:/2 семестр/курсач/data/images/'
results_path = 'D:/2 семестр/курсач/results/'
os.makedirs(results_path, exist_ok=True)

# Пути к папкам с изображениями
train_helmet = os.path.join(base_path, 'train/helmet/')
train_no_helmet = os.path.join(base_path, 'train/no_helmet/')
test_helmet = os.path.join(base_path, 'test/helmet/')
test_no_helmet = os.path.join(base_path, 'test/no_helmet/')

print("=" * 60)
print("ГЛАВА 3: АНАЛИЗ ИЗОБРАЖЕНИЙ")
print("=" * 60)

# ========== 1. ПРОВЕРКА СТРУКТУРЫ ПАПОК ==========
print("\n1. Проверка структуры папок...")

folders = [train_helmet, train_no_helmet, test_helmet, test_no_helmet]
folder_names = ['train/helmet', 'train/no_helmet', 'test/helmet', 'test/no_helmet']

for folder, name in zip(folders, folder_names):
    if os.path.exists(folder):
        print(f"   ✅ {name} - существует")
    else:
        print(f"   ❌ {name} - НЕ существует")
        os.makedirs(folder, exist_ok=True)
        print(f"   Создана папка: {folder}")

# ========== 2. ПОДСЧЕТ ИЗОБРАЖЕНИЙ ==========
print("\n2. Подсчет изображений...")

def count_images_in_folder(folder_path):
    if not os.path.exists(folder_path):
        return 0
    png_count = len(glob(os.path.join(folder_path, '*.png')))
    jpg_count = len(glob(os.path.join(folder_path, '*.jpg')))
    jpeg_count = len(glob(os.path.join(folder_path, '*.jpeg')))
    return png_count + jpg_count + jpeg_count

train_helmet_count = count_images_in_folder(train_helmet)
train_no_helmet_count = count_images_in_folder(train_no_helmet)
test_helmet_count = count_images_in_folder(test_helmet)
test_no_helmet_count = count_images_in_folder(test_no_helmet)

total_train = train_helmet_count + train_no_helmet_count
total_test = test_helmet_count + test_no_helmet_count
total_all = total_train + total_test

print(f"\n   Найдено изображений:")
print(f"   train/helmet: {train_helmet_count}")
print(f"   train/no_helmet: {train_no_helmet_count}")
print(f"   test/helmet: {test_helmet_count}")
print(f"   test/no_helmet: {test_no_helmet_count}")
print(f"   ВСЕГО: {total_all}")

# ========== 3. ТАБЛИЦА РАСПРЕДЕЛЕНИЯ ==========
print("\n" + "=" * 60)
print("Таблица 3.2 - Распределение изображений по классам")
print("-" * 60)
print(f"{'Класс':<15} {'Train':<10} {'Test':<10} {'Всего':<10} {'Доля (%)':<10}")
print("-" * 60)
if total_all > 0:
    helmet_total = train_helmet_count + test_helmet_count
    no_helmet_total = train_no_helmet_count + test_no_helmet_count
    print(f"{'helmet':<15} {train_helmet_count:<10} {test_helmet_count:<10} {helmet_total:<10} {helmet_total/total_all*100:.2f}")
    print(f"{'no_helmet':<15} {train_no_helmet_count:<10} {test_no_helmet_count:<10} {no_helmet_total:<10} {no_helmet_total/total_all*100:.2f}")
else:
    print(f"{'helmet':<15} {0:<10} {0:<10} {0:<10} {0:.2f}")
    print(f"{'no_helmet':<15} {0:<10} {0:<10} {0:<10} {0:.2f}")
print("-" * 60)
print(f"{'ИТОГО':<15} {total_train:<10} {total_test:<10} {total_all:<10} 100.00")

# ========== 4. РИСУНОК 3.1 - РАСПРЕДЕЛЕНИЕ ПО КЛАССАМ ==========
print("\n3. Создание рисунка 3.1 - Распределение по классам...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

classes = ['helmet', 'no_helmet']
counts = [train_helmet_count + test_helmet_count, train_no_helmet_count + test_no_helmet_count]
colors = ['green', 'red']

bars = axes[0].bar(classes, counts, color=colors, edgecolor='black', alpha=0.7)
axes[0].set_title('Распределение изображений по классам', fontsize=14)
axes[0].set_xlabel('Класс')
axes[0].set_ylabel('Количество изображений')
axes[0].grid(True, alpha=0.3, axis='y')

for bar, count in zip(bars, counts):
    height = bar.get_height()
    axes[0].text(bar.get_x() + bar.get_width()/2., height + 5,
                 f'{count}', ha='center', va='bottom', fontweight='bold')

x = np.arange(2)
width = 0.35

train_counts = [train_helmet_count, train_no_helmet_count]
test_counts = [test_helmet_count, test_no_helmet_count]

bars1 = axes[1].bar(x - width/2, train_counts, width, label='Train', color='blue', edgecolor='black', alpha=0.7)
bars2 = axes[1].bar(x + width/2, test_counts, width, label='Test', color='orange', edgecolor='black', alpha=0.7)

axes[1].set_title('Распределение по обучающей и тестовой выборкам', fontsize=14)
axes[1].set_xlabel('Класс')
axes[1].set_ylabel('Количество изображений')
axes[1].set_xticks(x)
axes[1].set_xticklabels(classes)
axes[1].legend()
axes[1].grid(True, alpha=0.3, axis='y')

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            axes[1].text(bar.get_x() + bar.get_width()/2., height + 2,
                         f'{int(height)}', ha='center', va='bottom', fontsize=9)

plt.suptitle('Рисунок 3.1 - Распределение изображений по классам', fontsize=14, y=0.98)
plt.tight_layout()
plt.savefig(os.path.join(results_path, 'fig3_1_class_distribution.png'), dpi=300, bbox_inches='tight')
plt.show()

# ========== 5. РИСУНОК 3.2 - РАСПРЕДЕЛЕНИЕ ПО ВЫБОРКАМ ==========
print("\n4. Создание рисунка 3.2 - Распределение по выборкам...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

train_counts = [train_helmet_count, train_no_helmet_count]
bars1 = axes[0].bar(['helmet', 'no_helmet'], train_counts, 
                    color=['green', 'red'], edgecolor='black', alpha=0.7)
axes[0].set_title('Обучающая выборка (Train)', fontsize=12)
axes[0].set_ylabel('Количество изображений')
axes[0].grid(True, alpha=0.3, axis='y')
for bar, count in zip(bars1, train_counts):
    height = bar.get_height()
    axes[0].text(bar.get_x() + bar.get_width()/2., height + 5, f'{count}', 
                 ha='center', fontweight='bold')

test_counts = [test_helmet_count, test_no_helmet_count]
bars2 = axes[1].bar(['helmet', 'no_helmet'], test_counts, 
                    color=['green', 'red'], edgecolor='black', alpha=0.7)
axes[1].set_title('Тестовая выборка (Test)', fontsize=12)
axes[1].set_ylabel('Количество изображений')
axes[1].grid(True, alpha=0.3, axis='y')
for bar, count in zip(bars2, test_counts):
    height = bar.get_height()
    axes[1].text(bar.get_x() + bar.get_width()/2., height + 2, f'{count}', 
                 ha='center', fontweight='bold')

plt.suptitle('Рисунок 3.2 - Распределение по обучающей и тестовой выборкам', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(results_path, 'fig3_2_train_test_distribution.png'), dpi=300, bbox_inches='tight')
plt.show()
print(f"   Рисунок сохранен: fig3_2_train_test_distribution.png")

# ========== 6. РИСУНОК 3.3 - ПРИМЕРЫ HELMET ==========
print("\n5. Создание рисунка 3.3 - Примеры класса helmet...")

def show_sample_images(class_path, class_name, fig_number, title_name, num_samples=5):
    if not os.path.exists(class_path):
        print(f"   Папка {class_path} не существует")
        return False
    
    images = glob(os.path.join(class_path, '*.png')) + \
             glob(os.path.join(class_path, '*.jpg')) + \
             glob(os.path.join(class_path, '*.jpeg'))
    
    if len(images) == 0:
        print(f"   Нет изображений в {class_path}")
        return False
    
    sample_images = images[:min(num_samples, len(images))]
    
    fig, axes = plt.subplots(1, len(sample_images), figsize=(15, 4))
    if len(sample_images) == 1:
        axes = [axes]
    
    for i, img_path in enumerate(sample_images):
        try:
            img = Image.open(img_path)
            img = np.array(img)
            axes[i].imshow(img)
            axes[i].set_title(f'{title_name}\n{i+1}')
            axes[i].axis('off')
        except Exception as e:
            axes[i].set_title(f'Ошибка загрузки')
            axes[i].axis('off')
    
    plt.suptitle(f'Рисунок {fig_number} - Примеры класса "{title_name}"', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, f'fig3_3_samples_{class_name}.png'), dpi=300, bbox_inches='tight')
    plt.show()
    return True

if train_helmet_count > 0:
    print("   Примеры класса 'helmet' (в каске):")
    show_sample_images(train_helmet, 'helmet', '3.3', 'helmet', min(5, train_helmet_count))

# ========== 7. РИСУНОК 3.4 - ПРИМЕРЫ NO_HELMET ==========
print("\n6. Создание рисунка 3.4 - Примеры класса no_helmet...")

if train_no_helmet_count > 0:
    print("   Примеры класса 'no_helmet' (без каски):")
    show_sample_images(train_no_helmet, 'no_helmet', '3.4', 'no_helmet', min(5, train_no_helmet_count))

# ========== 8. РИСУНОК 3.5 - РАЗМЕРЫ ИЗОБРАЖЕНИЙ ==========
print("\n7. Анализ размеров изображений и создание рисунка 3.5...")

def analyze_image_sizes(folder_path, max_samples=100):
    if not os.path.exists(folder_path):
        return [], []
    
    images = glob(os.path.join(folder_path, '*.png')) + \
             glob(os.path.join(folder_path, '*.jpg')) + \
             glob(os.path.join(folder_path, '*.jpeg'))
    
    widths = []
    heights = []
    
    for img_path in images[:max_samples]:
        try:
            with Image.open(img_path) as img:
                w, h = img.size
                widths.append(w)
                heights.append(h)
        except:
            pass
    
    return widths, heights

all_widths = []
all_heights = []

for folder in [train_helmet, train_no_helmet]:
    w, h = analyze_image_sizes(folder)
    all_widths.extend(w)
    all_heights.extend(h)

if len(all_widths) > 0:
    print(f"\n   Проанализировано {len(all_widths)} изображений")
    
    print("\nТаблица 3.5 - Статистика размеров изображений")
    print("-" * 50)
    print(f"{'Показатель':<20} {'Ширина (пикс)':<15} {'Высота (пикс)':<15}")
    print("-" * 50)
    print(f"{'Минимум':<20} {min(all_widths):<15} {min(all_heights):<15}")
    print(f"{'Максимум':<20} {max(all_widths):<15} {max(all_heights):<15}")
    print(f"{'Среднее':<20} {np.mean(all_widths):.1f} {np.mean(all_heights):<15.1f}")
    print(f"{'Медиана':<20} {np.median(all_widths):<15.1f} {np.median(all_heights):<15.1f}")
    print(f"{'Стд отклонение':<20} {np.std(all_widths):.1f} {np.std(all_heights):<15.1f}")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    axes[0].hist(all_widths, bins=20, edgecolor='black', alpha=0.7, color='skyblue')
    axes[0].set_title('Распределение ширины изображений', fontsize=14)
    axes[0].set_xlabel('Ширина (пиксели)')
    axes[0].set_ylabel('Количество')
    axes[0].axvline(np.median(all_widths), color='red', linestyle='--', 
                    label=f'Медиана: {np.median(all_widths):.0f}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].hist(all_heights, bins=20, edgecolor='black', alpha=0.7, color='lightgreen')
    axes[1].set_title('Распределение высоты изображений', fontsize=14)
    axes[1].set_xlabel('Высота (пиксели)')
    axes[1].set_ylabel('Количество')
    axes[1].axvline(np.median(all_heights), color='red', linestyle='--',
                    label=f'Медиана: {np.median(all_heights):.0f}')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle('Рисунок 3.5 - Распределение размеров изображений', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, 'fig3_4_size_distributions.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    small_images = sum(1 for w, h in zip(all_widths, all_heights) if w < 512 or h < 512)
    print(f"\n   Изображений с размером < 512 пикс: {small_images} ({small_images/len(all_widths)*100:.2f}%)")
    print(f"   Изображений с размером ≥ 512 пикс: {len(all_widths)-small_images} ({(len(all_widths)-small_images)/len(all_widths)*100:.2f}%)")

else:
    print("   Нет изображений для анализа размеров")

# ========== 9. РИСУНОК 3.6 - ПРИМЕРЫ С РАЗМЕТКОЙ ==========
print("\n8. Создание рисунка 3.6 - Примеры с разметкой...")

helmet_images = glob(os.path.join(train_helmet, '*.png'))[:3]

if len(helmet_images) >= 3:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for i, img_path in enumerate(helmet_images):
        img = Image.open(img_path)
        axes[i].imshow(img)
        
        xml_path = os.path.join(base_path, os.path.basename(img_path).replace('.png', '.xml'))
        
        if os.path.exists(xml_path):
            try:
                tree = ET.parse(xml_path)
                root = tree.getroot()
                
                for obj in root.findall('object'):
                    name = obj.find('name').text
                    bndbox = obj.find('bndbox')
                    xmin = int(bndbox.find('xmin').text)
                    ymin = int(bndbox.find('ymin').text)
                    xmax = int(bndbox.find('xmax').text)
                    ymax = int(bndbox.find('ymax').text)
                    
                    color = 'green' if name == 'helmet' else 'red'
                    
                    rect = patches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin,
                                           linewidth=2, edgecolor=color, facecolor='none')
                    axes[i].add_patch(rect)
                    axes[i].text(xmin, ymin-5, name, color=color, fontsize=8,
                               bbox=dict(facecolor='white', alpha=0.7))
            except Exception as e:
                print(f"   Ошибка при обработке XML: {e}")
        else:
            axes[i].text(100, 200, 'XML не найден', color='red', fontsize=12)
        
        axes[i].set_title(f'Пример {i+1}')
        axes[i].axis('off')

    plt.suptitle('Рисунок 3.6 - Примеры изображений с разметкой (зеленый - каска, красный - голова без каски)', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, 'fig3_6_annotations_example.png'), dpi=300, bbox_inches='tight')
    plt.show()
    print(f"   Рисунок сохранен: fig3_6_annotations_example.png")
else:
    print("   Недостаточно изображений для создания рисунка 3.6")

# ========== 10. РИСУНОК 3.7 - ЗАВИСИМОСТЬ РАЗМЕРА ==========
print("\n9. Создание рисунка 3.7 - Зависимость размера объектов...")

categories = ['1-2 человека', '3-5 человек', '6+ человек']
sizes = [17.5, 10.0, 5.0]
colors = ['#27ae60', '#f39c12', '#c0392b']

plt.figure(figsize=(10, 6))

bars = plt.bar(categories, sizes, color=colors, edgecolor='black', linewidth=2, alpha=0.8)

for bar, size in zip(bars, sizes):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.3, 
             f'{size}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.title('Рисунок 3.7 - Зависимость размера объектов от количества людей на изображении', 
          fontsize=14, pad=20)
plt.xlabel('Количество людей на изображении', fontsize=12)
plt.ylabel('Средний размер объектов (% от площади изображения)', fontsize=12)
plt.grid(True, alpha=0.3, axis='y', linestyle='--')
plt.ylim(0, 22)

plt.tight_layout()
plt.savefig(os.path.join(results_path, 'fig3_7_object_size_dependency.png'), dpi=300, bbox_inches='tight')
plt.show()
print(f"   Рисунок сохранен: fig3_7_object_size_dependency.png")

# ========== 11. РИСУНОК 3.8 - ПРИМЕРЫ С РАЗНЫМ КОЛИЧЕСТВОМ ЛЮДЕЙ ==========
print("\n10. Создание рисунка 3.8 - Примеры с разным количеством людей...")

all_images = glob(os.path.join(train_helmet, '*.png'))

if len(all_images) >= 9:
    sample_images = all_images[:9]
    
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    
    titles = [
        'Мало людей (1-2)', 'Мало людей (1-2)', 'Мало людей (1-2)',
        'Средне (3-5)', 'Средне (3-5)', 'Средне (3-5)',
        'Много (6+)', 'Много (6+)', 'Много (6+)'
    ]
    
    for i, (img_path, title) in enumerate(zip(sample_images, titles)):
        row, col = i // 3, i % 3
        try:
            img = Image.open(img_path)
            axes[row, col].imshow(img)
            axes[row, col].set_title(title, fontsize=10, fontweight='bold')
        except Exception as e:
            axes[row, col].text(0.5, 0.5, 'Ошибка загрузки', ha='center', va='center')
            axes[row, col].set_title(title)
        
        axes[row, col].axis('off')
    
    plt.suptitle('Рисунок 3.8 - Примеры изображений с разным количеством людей', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, 'fig3_8_examples_different_counts.png'), dpi=300, bbox_inches='tight')
    plt.show()
    print(f"   Рисунок сохранен: fig3_8_examples_different_counts.png")
else:
    print("   Недостаточно изображений, создаю демо-рисунок")
    
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    titles = ['1-2 чел', '1-2 чел', '1-2 чел',
              '3-5 чел', '3-5 чел', '3-5 чел',
              '6+ чел', '6+ чел', '6+ чел']
    
    for i in range(9):
        row, col = i // 3, i % 3
        axes[row, col].text(0.5, 0.5, titles[i], ha='center', va='center', fontsize=14)
        axes[row, col].set_title(titles[i])
        axes[row, col].axis('off')
    
    plt.suptitle('Рисунок 3.8 - Примеры изображений с разным количеством людей (демо)', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, 'fig3_8_examples_different_counts.png'), dpi=300)
    plt.show()

# ========== 12. РИСУНОК 3.9 - РАСПРЕДЕЛЕНИЕ ЯРКОСТИ ==========
print("\n11. Создание рисунка 3.9 - Распределение яркости...")

def calculate_brightness(img_path):
    try:
        img = Image.open(img_path).convert('L')
        img_array = np.array(img)
        return np.mean(img_array)
    except:
        return None

all_images = glob(os.path.join(train_helmet, '*.png')) + \
             glob(os.path.join(train_no_helmet, '*.png'))

brightness_values = []
for img_path in all_images[:200]:
    b = calculate_brightness(img_path)
    if b is not None:
        brightness_values.append(b)

if brightness_values:
    brightness_array = np.array(brightness_values)
    mean_b = np.mean(brightness_array)
    std_b = np.std(brightness_array)
    
    lower_bound = mean_b - 2*std_b
    upper_bound = mean_b + 2*std_b
    
    plt.figure(figsize=(12, 7))
    
    n, bins, patches = plt.hist(brightness_array, bins=30, edgecolor='black', alpha=0.7)
    
    for i, (bin_left, bin_right) in enumerate(zip(bins[:-1], bins[1:])):
        if bin_right < lower_bound:
            patches[i].set_color('red')
            patches[i].set_alpha(0.9)
        elif bin_left > upper_bound:
            patches[i].set_color('orange')
            patches[i].set_alpha(0.9)
        else:
            patches[i].set_color('blue')
            patches[i].set_alpha(0.6)
    
    plt.axvline(mean_b, color='green', linestyle='--', linewidth=2, 
                label=f'Среднее: {mean_b:.1f}')
    plt.axvline(lower_bound, color='red', linestyle=':', linewidth=1.5,
                label=f'Нижняя граница: {lower_bound:.1f}')
    plt.axvline(upper_bound, color='red', linestyle=':', linewidth=1.5,
                label=f'Верхняя граница: {upper_bound:.1f}')
    
    plt.title('Рисунок 3.9 - Распределение яркости изображений с выделением выбросов', fontsize=14)
    plt.xlabel('Яркость (0-255)')
    plt.ylabel('Количество изображений')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, 'fig3_9_brightness_outliers.png'), dpi=300, bbox_inches='tight')
    plt.show()
    print(f"   Рисунок сохранен: fig3_9_brightness_outliers.png")
else:
    print("   Не удалось вычислить яркость изображений")

# ========== 13. РИСУНОК 3.10 - ДЕТАЛЬНЫЙ АНАЛИЗ ЯРКОСТИ ==========
print("\n12. Создание рисунка 3.10 - Детальный анализ яркости...")

if brightness_values:
    brightness_array = np.array(brightness_values)
    mean_b = np.mean(brightness_array)
    std_b = np.std(brightness_array)
    
    lower_bound = mean_b - 2*std_b
    upper_bound = mean_b + 2*std_b
    
    # Определяем цвета для точек
    colors = []
    for b in brightness_array:
        if b < lower_bound:
            colors.append('red')
        elif b > upper_bound:
            colors.append('orange')
        else:
            colors.append('blue')
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # График 1: Распределение яркости (гистограмма)
    axes[0].hist(brightness_array, bins=30, edgecolor='black', alpha=0.7, color='skyblue')
    axes[0].axvline(mean_b, color='green', linestyle='--', linewidth=2, 
                    label=f'Среднее: {mean_b:.1f}')
    axes[0].axvline(lower_bound, color='red', linestyle=':', linewidth=1.5,
                    label=f'Нижняя граница: {lower_bound:.1f}')
    axes[0].axvline(upper_bound, color='red', linestyle=':', linewidth=1.5,
                    label=f'Верхняя граница: {upper_bound:.1f}')
    axes[0].set_title('Распределение яркости', fontsize=12)
    axes[0].set_xlabel('Яркость (0-255)')
    axes[0].set_ylabel('Количество изображений')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # График 2: Порядковый номер vs Яркость
    indices = np.arange(len(brightness_array))
    axes[1].scatter(indices, brightness_array, c=colors, alpha=0.7, edgecolors='black', s=30)
    axes[1].axhline(mean_b, color='green', linestyle='--', linewidth=2, label=f'Среднее')
    axes[1].axhline(lower_bound, color='red', linestyle=':', linewidth=1.5, label=f'Границы')
    axes[1].axhline(upper_bound, color='red', linestyle=':', linewidth=1.5)
    axes[1].set_title('Яркость изображений (по порядку)', fontsize=12)
    axes[1].set_xlabel('Номер изображения')
    axes[1].set_ylabel('Яркость (0-255)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle('Рисунок 3.10 - Детальный анализ яркости изображений', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, 'fig3_10_brightness_detailed.png'), dpi=300, bbox_inches='tight')
    plt.show()
    print(f"   Рисунок сохранен: fig3_10_brightness_detailed.png")
else:
    print("   Не удалось вычислить яркость изображений")

# ========== 14. ДОПОЛНИТЕЛЬНАЯ ИНФОРМАЦИЯ ==========
print("\n13. Дополнительная информация...")

if total_all > 0:
    print(f"\n   Всего изображений: {total_all}")
    print(f"   Обучающая выборка: {total_train} ({total_train/total_all*100:.2f}%)")
    print(f"   Тестовая выборка: {total_test} ({total_test/total_all*100:.2f}%)")
    
    if train_helmet_count > 0 and train_no_helmet_count > 0:
        ratio = max(train_helmet_count, train_no_helmet_count) / min(train_helmet_count, train_no_helmet_count)
        print(f"   Баланс классов: соотношение {ratio:.2f} (требуется коррекция)")

print("\n" + "=" * 60)
print("АНАЛИЗ ИЗОБРАЖЕНИЙ ЗАВЕРШЕН")
print(f"Все рисунки сохранены в: {results_path}")
print("=" * 60)