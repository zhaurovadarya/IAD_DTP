import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination
import warnings
warnings.filterwarnings("ignore")

DF_path = "df_rules.csv"

print("Loading data...")
data = pd.read_csv(DF_path, low_memory=False)
print("df_rules:", data.shape)
data['severity_score'] = 2 * data['fatality_count'] + data['injury_count']

def find_candidate_features(df):
    candidates = []
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_numeric_dtype(s):
            if (s.fillna(0) > 0).any():
                candidates.append(col)
        else:
            s_str = s.astype(str).fillna("").str.strip()
            if (s_str != "").any():
                candidates.append(col)
    print(f"\nВсего найдено кандидатов с ненулевыми значениями: {len(candidates)}")
    for c in candidates:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s):
            print(f"{c} (числовая): ненулевых значений = {(s.fillna(0) > 0).sum()}")
        else:
            print(f"{c} (строковая): непустых значений = {(s_str != '').sum()}")
    return candidates
all_candidates = find_candidate_features(data)

severity = "severity_class" if "severity_class" in data.columns else None
if severity is None:
    raise ValueError("severity_class не найдена")
mode = "analysis"  # анализ влияния факторов
exclude = ["fatality_count", "injury_count", "severity_score"]
parents = [c for c in all_candidates if c not in exclude and c != severity]
print("\nФакторы (parents) для сети:", parents)
df_bn = data[parents + [severity]].copy()

def to_binary_series(s):
    if pd.api.types.is_numeric_dtype(s):
        return s.fillna(0).apply(lambda x: 1 if x > 0 else 0).astype(int)
    else:
        return s.astype(str).fillna("unknown")

for col in parents:
    if pd.api.types.is_numeric_dtype(df_bn[col]):
        print(f"Binarizing {col} ...")
        df_bn[col] = to_binary_series(df_bn[col])
    else:
        df_bn[col] = df_bn[col].astype(str).fillna("unknown")

df_bn[severity] = df_bn[severity].astype(str).fillna("unknown")
state_names = {col: sorted(df_bn[col].dropna().unique().tolist()) for col in df_bn.columns}
print("\nUnique states per variable:")
for col, vals in state_names.items():
    print(f"{col}: {vals}")

edges = [(p, severity) for p in parents] # cтруктура сети
print("\nNetwork edges:", edges)

model = DiscreteBayesianNetwork(edges)
print("\nTraining Bayesian network...")
model.fit(df_bn, estimator=BayesianEstimator, prior_type="BDeu", equivalent_sample_size=5)
print("✔ Training finished.")

# визуализация
G = nx.DiGraph()
G.add_edges_from(edges)
plt.figure(figsize=(10, 6))
pos = nx.spring_layout(G, seed=42)
nx.draw(G, pos, with_labels=True, node_size=2400, node_color="lightblue", font_size=8, arrowsize=16)
plt.title("Bayesian Network structure")
plt.tight_layout()
plt.savefig("bayes_network.png", dpi=200)
plt.show()
print("Saved: bayes_network.png")

infer = VariableElimination(model) # анализ важности факторов
factors = parents
importance_report = []

for f in factors:
    cpd_f = model.get_cpds(f)
    if cpd_f is not None and f in cpd_f.state_names:
        states = cpd_f.state_names[f]
    else:
        states = df_bn[f].unique()

    impact = []
    for state in states:
        q = infer.query(variables=[severity], evidence={f: state})
        probs = np.array(q.values).flatten()
        entropy = -np.sum(probs * np.log2(probs + 1e-9))
        impact.append(entropy)

    importance_report.append({
        'factor': f,
        'states': list(states),
        'entropy_per_state': impact,
        'mean_entropy': np.mean(impact) })
report_df = pd.DataFrame(importance_report)
report_df = report_df.sort_values('mean_entropy')
report_df.to_csv('Bayes_importance.csv', index=False, encoding='utf-8-sig')
print("Saved factor importance report -> Bayes_importance.csv")
df = pd.read_csv("Bayes_importance.csv")
df = df.sort_values('mean_entropy')
plt.figure(figsize=(10,6))
plt.barh(df['factor'], df['mean_entropy'], color='skyblue')
plt.xlabel('Mean Entropy')
plt.title('Информативность факторов для severity_class')
plt.gca().invert_yaxis()
plt.show()
