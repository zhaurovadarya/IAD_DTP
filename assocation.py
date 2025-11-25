from mlxtend.frequent_patterns import apriori, association_rules
from tqdm import tqdm
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("df_rules.csv", low_memory=False)
razmet_df = pd.read_csv("razmetPF.csv")
print(f"df_rules: {df.shape[0]} строк, {df.shape[1]} столбцов")
print(f"Разметка ПДД: {razmet_df.shape[0]} пунктов")

exclude_cols = ['latitude', 'longitude', 'severity_score', 'fatality_count', 'injury_count'] #бинаризация признаков
features = [col for col in df.columns if col not in exclude_cols and col != 'severity_class']

df_bin = df[features].fillna(0).astype(bool)
severity_class = df['severity_class'].fillna('unknown').astype(str) # добавление бин severity_class
severity_bin = pd.get_dummies(severity_class, prefix='severity_class')
df_bin = pd.concat([df_bin, severity_bin], axis=1)
frequent_itemsets = apriori(df_bin, min_support=0.01, use_colnames=True)
print(f"🔍 Найдено {len(frequent_itemsets)} частых наборов признаков")

rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0) # генерация ассоциативных правил
rules = rules[rules['consequents'].apply(lambda x: any('severity_class' in item for item in x))] # правила, где consequents связаны с severity_class
rules = rules.sort_values(by="lift", ascending=False).reset_index(drop=True)
print(f"🔹 Сгенерировано правил: {len(rules)}")

# преобразование antecedents и consequents в списки
rules['antecedents_list'] = rules['antecedents'].apply(lambda x: list(x))
rules['consequents_list'] = rules['consequents'].apply(lambda x: list(x))
proposals = [] # сопоставление по подстрокам / леммам
for idx, rule in tqdm(rules.iterrows(), total=len(rules), desc="Сопоставление правил"):
    antecedents = rule['antecedents_list']

    for _, pdd_row in razmet_df.iterrows():
        related_factors = str(pdd_row['related_factors']).split(', ')
        matched_factors = []

        for a in antecedents:
            for rf in related_factors:
                if a.lower() in rf.lower() or rf.lower() in a.lower():
                    matched_factors.append(rf)

        if matched_factors:
            proposals.append({
                "rule_index": idx,
                "antecedents": rule['antecedents'],
                "consequents": rule['consequents'],
                "lift": rule['lift'],
                "pdd_id": pdd_row['pdd_id'],
                "pdd_text": pdd_row['pdd_text'],
                "themes": pdd_row['themes'],
                "matched_factors": ", ".join(set(matched_factors))
            })

proposals_df = pd.DataFrame(proposals)
proposals_df.to_csv("PDD_coincidence.csv", index=False, encoding="utf-8-sig")
print(f"💾 Сопоставление с ПДД выполнено, {len(proposals_df)} совпадений сохранено")

# средние значения для метрик
mean_support = rules['support'].mean()
mean_confidence = rules['confidence'].mean()
mean_lift = rules['lift'].mean()
print(f"Средний Support: {mean_support:.3f}")
print(f"Средний Confidence: {mean_confidence:.3f}")
print(f"Средний Lift: {mean_lift:.3f}")

rules.to_csv("assocPR.csv", index=False, encoding="utf-8-sig")
print("💾 Ассоциативные правила сохранены в assocPR.csv")

# визуализация
factor_counts = proposals_df['matched_factors'].str.split(', ').explode().value_counts()
top_factors = factor_counts.head(10).index.tolist()
pdd_counts = proposals_df.groupby('pdd_id').size().sort_values(ascending=False)
top_pdds = pdd_counts.head(10).index.tolist()

subset = proposals_df[ # фильтрация
    proposals_df['matched_factors'].str.split(', ').apply(
        lambda x: any(f in top_factors for f in x))]
subset = subset[subset['pdd_id'].isin(top_pdds)]

matrix = pd.DataFrame(0.0, index=top_pdds, columns=top_factors)
pair_counts = {(p, f): 0 for p in top_pdds for f in top_factors}

for _, row in subset.iterrows():
    pdd = row['pdd_id']
    factors = row['matched_factors'].split(', ')
    for f in factors:
        if f in top_factors:
            pair_counts[(pdd, f)] += 1
pdd_total = {pdd: 0 for pdd in top_pdds}
for (pdd, f), count in pair_counts.items():
    pdd_total[pdd] += count
for (pdd, f), count in pair_counts.items():
    if pdd_total[pdd] > 0:
        matrix.loc[pdd, f] = count / pdd_total[pdd]
    else:
        matrix.loc[pdd, f] = 0.0
plt.figure(figsize=(14, 7))
sns.heatmap(matrix, annot=True, fmt=".2f", cmap='YlGnBu')
plt.xlabel("Факторы", fontsize=14)
plt.ylabel("Пункты ПДД", fontsize=14)
plt.title("Сила связи факторов с пунктами ПДД", fontsize=14)
plt.tight_layout()
plt.show()

theme_counts = proposals_df['themes'].str.split(', ').explode().value_counts() #
plt.figure(figsize=(14, 6))
theme_counts.head(20).plot(kind='bar', color='lightgreen')
plt.ylabel("Количество совпадений", fontsize=14)
plt.xlabel("Темы", fontsize=14)
plt.title("Распределение совпадений по темам", fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

