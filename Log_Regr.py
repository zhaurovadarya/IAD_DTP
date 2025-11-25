import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.utils import resample
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

def main():
    engine = create_engine("postgresql+psycopg2://razrab:puk5@localhost:5432/DTP")

    query = """
    WITH dpt_agg AS (
        SELECT 
            n.id_uchastock,
            DATE(n.date_time) AS date_,
            CASE WHEN COUNT(a.emtp_number) > 0 THEN 1 ELSE 0 END AS dpt_occurred
        FROM nabludenie_gor n
        LEFT JOIN link_dtp_gor_uchastok l ON n.id_uchastock = l.id_uchastock
        LEFT JOIN accident a ON l.emtp_number = a.emtp_number
        GROUP BY n.id_uchastock, DATE(n.date_time)
    )
    SELECT 
        n.id_uchastock,
        n.date_time,
        n.density,
        n.speed,
        n.average_distance_between_objects,
        ai.weather_condition,
        ai.road_condition,
        ai.lighting_condition,
        ai.traffic_rules,
        p.driving_experience,
        p.alco,
        v.g_v,
        v.t_ts,
        ad.street_category,
        da.dpt_occurred
    FROM nabludenie_gor n
    LEFT JOIN link_dtp_gor_uchastok l ON n.id_uchastock = l.id_uchastock
    LEFT JOIN accident a ON l.emtp_number = a.emtp_number
    LEFT JOIN accident_influence ai ON a.influence_id = ai.influence_id
    LEFT JOIN addres ad ON a.address_id = ad.address_id
    LEFT JOIN participants p ON a.emtp_number = p.emtp_number
    LEFT JOIN vehicles v ON a.emtp_number = v.emtp_number
    LEFT JOIN dpt_agg da ON n.id_uchastock = da.id_uchastock AND DATE(n.date_time) = da.date_;
    
    """

    data = pd.read_sql(query, engine)
    data['dpt_occurred'] = data['dpt_occurred'].fillna(0)

    num_cols = ['density', 'speed','average_distance_between_objects','driving_experience','alco','g_v']
    for col in num_cols:
        data[col] = data[col].fillna(data[col].median())
    categorical_cols = ['street_category', 'road_condition', 'weather_condition', 'lighting_condition', 'traffic_rules','t_ts']
    data = pd.get_dummies(data, columns=categorical_cols, drop_first=False)

    X = data.drop(columns=['dpt_occurred', 'id_uchastock', 'date_time'])
    y = data['dpt_occurred']

    scaler = StandardScaler()
    X[['density', 'speed','average_distance_between_objects','driving_experience','alco','g_v']] = scaler.fit_transform(X[['density','speed','average_distance_between_objects','driving_experience','alco','g_v']]) #масштабирование числ признаков
    cols_to_remove = [
        col for col in X.columns if any(c in col for c in ['weather_condition', 'lighting_condition',
                                                           'road_condition', 'traffic_rules','t_ts'])]
    print(f"Удаляем признаки, связанные с утечкой данных: {cols_to_remove}")
    X = X.drop(columns=cols_to_remove)
    n_synthetic = int(y.sum() * 2)  # добавление синт данных в 2х "0"
    synthetic_data = pd.DataFrame({
        'density': np.random.normal(loc=X['density'].mean(), scale=X['density'].std(), size=n_synthetic),
        'speed': np.random.normal(loc=X['speed'].mean(), scale=X['speed'].std(), size=n_synthetic),
        'average_distance_between_objects': np.random.normal(loc=X['average_distance_between_objects'].mean(), scale=X['average_distance_between_objects'].std(), size=n_synthetic),
        'driving_experience': np.random.normal(loc=X['driving_experience'].mean(), scale=X['driving_experience'].std(), size=n_synthetic),
        'alco': np.random.normal(loc=X['alco'].mean(), scale=X['alco'].std(), size=n_synthetic),
        'g_v': np.random.normal(loc=X['g_v'].mean(), scale=X['g_v'].std(), size=n_synthetic),})
    for col in X.columns: # категориальные сделать случайными
        if col not in ['density','speed','average_distance_between_objects','driving_experience','alco','g_v']:
            synthetic_data[col] = np.random.randint(0,2,size=n_synthetic)
    synthetic_target = pd.Series([0]*n_synthetic, name='dpt_occurred')
    data_bal = pd.concat([pd.concat([X, y], axis=1), pd.concat([synthetic_data, synthetic_target], axis=1)], ignore_index=True)

    majority = data_bal[data_bal.dpt_occurred == 1] # upsampling
    minority = data_bal[data_bal.dpt_occurred == 0]
    minority_upsampled = resample(
        minority, replace=True, n_samples=len(majority), random_state=42)
    data_bal = pd.concat([majority, minority_upsampled])

    X = data_bal.drop('dpt_occurred', axis=1) # обновление после балансировки
    y = data_bal['dpt_occurred']
    X['density'] += np.random.normal(0, 0.5, size=len(X)) # добавление шума
    X['speed'] += np.random.normal(0, 0.5, size=len(X))
    X['average_distance_between_objects'] += np.random.normal(0, 0.5, size=len(X))
    X['driving_experience'] += np.random.normal(0, 0.5, size=len(X))
    X['alco'] += np.random.normal(0, 0.5, size=len(X))
    X['g_v'] += np.random.normal(0, 0.5, size=len(X))
    print("Распределение классов после балансировки и синтетики:")
    print(y.value_counts(normalize=True))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = LogisticRegression(max_iter=1000) #обучение лог регр на синт. выборке
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\nОтчет по классификации для городских наблюдений:")
    class_report = classification_report(y_test, y_pred)
    print(class_report)

    print("\nМатрица ошибок:")
    conf_matrix = confusion_matrix(y_test, y_pred)
    print(conf_matrix)

    roc_auc_val = roc_auc_score(y_test, y_prob)
    print("\nROC AUC:", roc_auc_val)

    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc_val:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve для городских наблюдений')
    plt.legend()

    roc_path = "roc_city.png"
    plt.savefig(roc_path, dpi=200, bbox_inches='tight')
    plt.show()
    plt.close()

    X_real = X.iloc[:len(data)]
    y_real = y.iloc[:len(data)]
    model_real = LogisticRegression(max_iter=1000)
    model_real.fit(X_real, y_real)

    feature_importance_real = pd.DataFrame({
        'feature': X_real.columns,
        'coef': model_real.coef_[0]
    }).sort_values(by='coef', key=abs, ascending=False)

    print("\nВажность признаков по городским наблюдениям:")
    print(feature_importance_real)

    plt.figure(figsize=(15, 6))
    sns.barplot(x='coef', y='feature', data=feature_importance_real, palette='coolwarm')
    plt.title('Коэффициенты логистической регрессии, влияющих на возникновение ДТП по городским наблюдениям', fontsize=14)
    plt.xlabel('Коэффициент логистической регрессии', labelpad=15, fontsize=14)
    plt.ylabel('Признак', labelpad=15, fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()

    coef_plot_path = "coef_importance_city.png"
    plt.savefig(coef_plot_path, dpi=200, bbox_inches='tight')
    plt.show()
    plt.close()

    coef_df = pd.DataFrame({
        'Признак': X_real.columns,
        'Коэффициент': model_real.coef_[0],
        'Отношение шансов (e^coef)': np.exp(model_real.coef_[0])
    }).sort_values(by='Коэффициент', ascending=False)

    print("\nИнтерпретация коэффициентов модели по городским наблюдениям:")
    print(coef_df)

    for i, row in coef_df.iterrows():
        effect = "повышает" if row['Коэффициент'] > 0 else "снижает"
        print(
            f"!Признак '{row['Признак']}' {effect} вероятность ДТП примерно в {row['Отношение шансов (e^coef)']:.2f} раз")

    plt.figure(figsize=(8, 6))
    sns.histplot(y_prob, bins=15, kde=True, alpha=0.6, edgecolor='black')
    plt.xlim(0, 1)
    plt.xticks(np.linspace(0, 1, 11))
    plt.xlabel('Вероятность ДТП', fontsize=12)
    plt.ylabel('Количество участков', fontsize=12)
    plt.title('Диаграмма распределения предсказанных вероятностей ДТП по городским наблюдениям', fontsize=14)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    prob_path = "prob_distribution_city.png"
    plt.savefig(prob_path, dpi=200, bbox_inches='tight')
    plt.show()
    plt.close()

    return {
    "log_city": {
         "classification_report": class_report,
        "conf_matrix": conf_matrix,
        "roc_auc": roc_auc_score(y_test, y_prob),
        "roc_plot": roc_path,
        "prob_dist_plot": prob_path,
        "log_coef": feature_importance_real,
        "coef_df": coef_df,
        "coef_plot": coef_plot_path
    },
    "log_coef": feature_importance_real
}

if __name__ == "__main__":
    main()


