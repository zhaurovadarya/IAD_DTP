import pandas as pd
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("df_rules.csv", low_memory=False)
razmet = pd.read_csv("razmetPF.csv")
print(df.shape, razmet.shape)

# определение нечетких входов
weather = ctrl.Antecedent(np.arange(0, 2, 1), 'weather_bad')
lighting = ctrl.Antecedent(np.arange(0, 2, 1), 'bad_lighting')
road = ctrl.Antecedent(np.arange(0, 2, 1), 'road_slippery')
vehicle = ctrl.Antecedent(np.arange(0, 2, 1), 'vehicle_heavy')
severity = ctrl.Consequent(np.arange(0, 101, 1), 'severity_score')

# нечеткие множества
severity['low'] = fuzz.trimf(severity.universe, [0, 0, 25])
severity['medium'] = fuzz.trimf(severity.universe, [20, 50, 75])
severity['high'] = fuzz.trimf(severity.universe, [60, 100, 100])
weather['good'] = fuzz.trimf(weather.universe, [0, 0, 1])
weather['bad'] = fuzz.trimf(weather.universe, [0, 1, 1])
lighting['good'] = fuzz.trimf(lighting.universe, [0, 0, 1])
lighting['bad'] = fuzz.trimf(lighting.universe, [0, 1, 1])
road['good'] = fuzz.trimf(road.universe, [0, 0, 1])
road['slippery'] = fuzz.trimf(road.universe, [0, 1, 1])
vehicle['light'] = fuzz.trimf(vehicle.universe, [0, 0, 1])
vehicle['heavy'] = fuzz.trimf(vehicle.universe, [0, 1, 1])

# нечеткие правила
rules = [
    ctrl.Rule(weather['bad'] & road['slippery'], severity['high']),
    ctrl.Rule(weather['bad'] & lighting['bad'], severity['high']),
    ctrl.Rule(road['slippery'] & lighting['bad'], severity['high']),
    ctrl.Rule(vehicle['heavy'] & road['slippery'], severity['medium']),
    ctrl.Rule(weather['good'] & road['good'], severity['low']),
    ctrl.Rule(weather['bad'], severity['medium']),
    ctrl.Rule(road['slippery'], severity['medium']),
    ctrl.Rule(lighting['bad'], severity['medium']),
    ctrl.Rule(vehicle['heavy'] & (weather['bad'] | lighting['bad']), severity['medium'])]

system = ctrl.ControlSystem(rules)
sim = ctrl.ControlSystemSimulation(system)

def compute_severity(row): # расчет тяжести
    factors_str = str(row.get("related_factors", ""))
    factors = [f.strip().lower() for f in factors_str.split(",") if f.strip()] if factors_str.lower() != 'nan' else []
    sim.input['weather_bad'] = int(any("снег" in f or "дождь" in f or "туман" in f for f in factors))
    sim.input['bad_lighting'] = int(any("темно" in f or "фонар" in f for f in factors))
    sim.input['road_slippery'] = int(any("скользко" in f or "гололед" in f for f in factors))
    sim.input['vehicle_heavy'] = int(any("грузовик" in f or "автобус" in f for f in factors))
    try:
        sim.compute()
        return sim.output['severity_score']
    except:
        return np.nan

razmet['severity_score_fuzzy'] = razmet.apply(compute_severity, axis=1)

bins = [0, 20, 40, 100]   # границы на основе диапазона тяжести
labels = ['Низкая', 'Средняя', 'Тяжелая']
razmet['severity_category'] = pd.cut(
    razmet['severity_score_fuzzy'],
    bins=bins,
    labels=labels,
    include_lowest=True)

print("Описание severity_score_fuzzy:")
print(razmet['severity_score_fuzzy'].describe())
print("\nКоличество статей по категориям:")
print(razmet['severity_category'].value_counts())

# визуализация
sns.set(style="whitegrid")
plt.figure(figsize=(10,6))
category_counts = razmet['severity_category'].value_counts().reindex(labels, fill_value=0)

colors = ['#4daf4a', '#ffbf00', '#e41a1c']
bars = plt.bar(category_counts.index, category_counts.values, color=colors, edgecolor='black')
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 0.5, str(height), ha='center', fontsize=14)
plt.title('Распределение значений тяжести к статьям ПДД', fontsize=16)
plt.xlabel('Категория тяжести', fontsize=16)
plt.ylabel('Количество статей', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.ylim(0, category_counts.max() * 1.2)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
razmet.to_csv("Fuzzy_PDD.csv", index=False, encoding="utf-8-sig")
print("Сохранено: Fuzzy_PDD.csv")



