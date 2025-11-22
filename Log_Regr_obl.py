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
    WITH obl_agg AS (
        SELECT
            id_station,
            SUM(outward) AS outward,
            SUM(return_) AS return_
        FROM nabludenie_obl
        GROUP BY id_station
    )
    SELECT
        a.emtp_number,
        s.id_station,
        s.uchastock_name_obl,
        s.region_uchastock,
        oa.outward,
        oa.return_,
        ai.weather_condition,
        ai.road_condition,
        ai.lighting_condition,
        ai.traffic_rules,
        ad.street_category,
        CASE 
            WHEN l.id_link_obl IS NOT NULL THEN 1
            ELSE 0
        END AS dpt_occurred
    FROM accident a
    CROSS JOIN station s
    LEFT JOIN link_dtp_obl_uchastok l
        ON a.emtp_number = l.emtp_number
       AND s.id_station = l.id_station
    LEFT JOIN obl_agg oa
        ON s.id_station = oa.id_station
    LEFT JOIN accident_influence ai
        ON a.influence_id = ai.influence_id
    LEFT JOIN addres ad
        ON a.address_id = ad.address_id;
    """
    data = pd.read_sql(query, engine)
    data['dpt_occurred'] = data['dpt_occurred'].fillna(0)
    data['outward'] = data['outward'].fillna(0)
    data['return_'] = data['return_'].fillna(0)

    categorical_cols = ['street_category', 'road_condition', 'weather_condition', 'lighting_condition', 'traffic_rules']
    data = pd.get_dummies(data, columns=categorical_cols, drop_first=False)
    X = data.drop(columns=['dpt_occurred', 'emtp_number', 'id_station'])
    y = data['dpt_occurred']


    text_cols = X.select_dtypes(include='object').columns
    print(f"Удаляем текстовые признаки: {list(text_cols)}")
    X = X.drop(columns=text_cols)

    scaler = StandardScaler() #масштабирование числ признаков
    X[['outward', 'return_']] = scaler.fit_transform(X[['outward', 'return_']])

    n_synthetic = int(y.sum() * 2)  # добавление синт данных в 2х "0"
    synthetic_data = pd.DataFrame({
        'outward': np.random.normal(loc=X['outward'].mean(), scale=X['outward'].std(), size=n_synthetic),
        'return_': np.random.normal(loc=X['return_'].mean(), scale=X['return_'].std(), size=n_synthetic),})
    for col in X.columns:
        if col not in ['outward','return_']:
            synthetic_data[col] = np.random.randint(0,2,size=n_synthetic)
    synthetic_target = pd.Series([0]*n_synthetic, name='dpt_occurred')
    data_bal = pd.concat([pd.concat([X, y], axis=1), pd.concat([synthetic_data, synthetic_target], axis=1)], ignore_index=True)


    majority = data_bal[data_bal.dpt_occurred == 1] # upsampling
    minority = data_bal[data_bal.dpt_occurred == 0]
    minority_upsampled = resample(minority, replace=True, n_samples=len(majority), random_state=42)
    data_bal = pd.concat([majority, minority_upsampled])

    X = data_bal.drop('dpt_occurred', axis=1) # обновление после балансировки
    y = data_bal['dpt_occurred']
    X['outward'] += np.random.normal(0, 0.5, size=len(X)) # добавление шума
    X['return_'] += np.random.normal(0, 0.5, size=len(X))
    print("Распределение классов после балансировки и синтетики:")
    print(y.value_counts(normalize=True))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # --- 1. ОТЧЁТ ПО КЛАССИФИКАЦИИ ---
    print("\nОтчёт по классификации для областных наблюдений:")
    class_report = classification_report(y_test, y_pred)
    print(class_report)

    # --- 2. МАТРИЦА ОШИБОК ---
    print("\nМатрица ошибок:")
    conf_matrix = confusion_matrix(y_test, y_pred)
    print(conf_matrix)

    # --- 3. ROC AUC ---
    roc_auc_val = roc_auc_score(y_test, y_prob)
    print("\nROC AUC:", roc_auc_val)

    # --- 4. ROC-кривая ---
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc_val:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve для областных наблюдений')
    plt.legend()

    roc_path = "roc_obl.png"
    plt.savefig(roc_path, dpi=200, bbox_inches='tight')
    plt.show()
    plt.close()

    # --- 5. Важность признаков на реальных данных ---
    X_real = X.iloc[:len(data)]
    y_real = y.iloc[:len(data)]
    model_real = LogisticRegression(max_iter=1000)
    model_real.fit(X_real, y_real)

    feature_importance_real = pd.DataFrame({
        'feature': X_real.columns,
        'coef': model_real.coef_[0]
    }).sort_values(by='coef', key=abs, ascending=False)
    feature_importance_real = feature_importance_real[feature_importance_real['coef'].abs() >= 0.1]

    print("\nВажность признаков по областным наблюдениям:")
    print(feature_importance_real)


    # --- 7. Коэффициенты + отношение шансов на реальных данных ---
    coef_df = pd.DataFrame({
        'Признак': X_real.columns,
        'Коэффициент': model_real.coef_[0],
        'Отношение шансов (e^coef)': np.exp(model_real.coef_[0])
    })

    # --- сортировка по абсолютным значениям ---
    coef_df = coef_df.reindex(coef_df['Коэффициент'].abs().sort_values(ascending=False).index)

    # --- ТОП-10 значимых в консоли ---
    top_10 = coef_df.head(10)
    print("\nТОП-10 значимых признаков по областным дорогам:")
    print(top_10)

    # --- интерпретация (только значимые >0.1) ---
    print("\nИнтерпретация влияния признаков на вероятность ДТП по областным дорогам:\n")
    for i, row in coef_df.iterrows():
        if abs(row['Коэффициент']) < 0.1:
            continue
        effect = "повышает" if row['Коэффициент'] > 0 else "снижает"
        print(
            f"• Признак '{row['Признак']}' {effect} вероятность ДТП примерно в {row['Отношение шансов (e^coef)']:.2f} раз")

    # --- график значимых признаков ---
    significant_features = coef_df[coef_df['Коэффициент'].abs() >= 0.1]
    plt.figure(figsize=(15, 6))
    sns.barplot(
        x='Коэффициент',
        y='Признак',
        data=significant_features,
        palette='coolwarm'
    )
    plt.axvline(0, color='black', linestyle='--')
    plt.title('Коэффициенты признаков логистической регрессии, влияющих на возникновение ДТП по областным наблюдениям')
    plt.xlabel('Коэффициент логистической регрессии', labelpad=15, fontsize=14)
    plt.ylabel('Признак', labelpad=15, fontsize=14)
    plt.xticks(fontsize=12, rotation=0)
    plt.yticks(fontsize=12)
    plt.tight_layout()

    coef_plot_path = "coef_importance_obl.png"
    plt.savefig(coef_plot_path, dpi=200, bbox_inches='tight')
    plt.show()
    plt.close()

    # --- 8. Диаграмма распределения вероятностей ---
    plt.figure(figsize=(8, 6))
    sns.histplot(y_prob, bins=15, kde=True, alpha=0.6, edgecolor='black')
    plt.xlim(0, 1)
    plt.xticks(np.linspace(0, 1, 11))
    plt.xlabel('Вероятность ДТП', fontsize=12)
    plt.ylabel('Количество участков', fontsize=12)
    plt.title('Диаграмма распределения предсказанных вероятностей ДТП по областным наблюдениям', fontsize=14)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    prob_path = "prob_distribution_obl.png"
    plt.savefig(prob_path, dpi=200, bbox_inches='tight')
    plt.show()
    plt.close()

    return {
        "log_region": {
            "classification_report": class_report,
            "conf_matrix": conf_matrix,
            "roc_auc": roc_auc_score(y_test, y_prob),
            "roc_plot": roc_path,
            "prob_dist_plot": prob_path,
            "log_coef": feature_importance_real,  # важность на реальных данных
            "coef_df": coef_df,  # коэффициенты + отношение шансов
            "coef_plot": coef_plot_path  # путь к PNG с коэффициентами
        },
        "log_coef": feature_importance_real
    }


if __name__ == "__main__":
    main()