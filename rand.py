import pandas as pd
from sqlalchemy import create_engine
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import folium
from folium.plugins import HeatMap, MarkerCluster
import warnings
warnings.filterwarnings("ignore")


def main():
    engine = create_engine("postgresql+psycopg2://razrab:puk5@localhost:5432/DTP")
    query = """
    SELECT DISTINCT ON (a.emtp_number)
        a.emtp_number,
        a.date_,
        a.time_,
        a.accident_type,
        a.fatality_count,
        a.injury_count,
        a.latitude,
        a.longitude,
        ai.weather_condition,
        ai.road_condition,
        ai.lighting_condition,
        ad.district,
        ad.street_category,
        p.alco,
        p.driving_experience,
        v.g_v,
        v.t_ts,
        v.technical_faults,
        -- Факторы и дефициты объединяем для последующего разложения в признаки
        STRING_AGG(DISTINCT d.ndu, ',') AS deficiencies,
        STRING_AGG(DISTINCT f.factor, ',') AS factors
    FROM accident a
    JOIN addres ad ON a.address_id = ad.address_id
    JOIN accident_influence ai ON a.influence_id = ai.influence_id
    LEFT JOIN influence_deficiency idf ON ai.influence_id = idf.influence_id
    LEFT JOIN deficiencies d ON idf.deficiency_id = d.deficiency_id
    LEFT JOIN influence_factor ifa ON ai.influence_id = ifa.influence_id
    LEFT JOIN factors f ON ifa.factor_id = f.factor_id
    JOIN (
        SELECT DISTINCT ON (emtp_number) 
            emtp_number, alco, driving_experience
        FROM participants
        WHERE participant_category = 'Водитель'
        ORDER BY emtp_number, driving_experience DESC
    ) p ON a.emtp_number = p.emtp_number
    JOIN (
        SELECT DISTINCT ON (emtp_number) 
            emtp_number, g_v, t_ts, technical_faults
        FROM vehicles
        ORDER BY emtp_number, g_v
    ) v ON a.emtp_number = v.emtp_number
    GROUP BY
        a.emtp_number, a.date_, a.time_, a.accident_type,
        a.latitude, a.longitude,
        ai.weather_condition, ai.road_condition, ai.lighting_condition,ad.street_category,
        ad.district, p.alco, p.driving_experience, v.g_v, v.t_ts, v.technical_faults;
    """
    data = pd.read_sql(query, engine)
    geo_raw = data[['latitude', 'longitude', 'fatality_count', 'injury_count', 'accident_type', 'district']].copy()
    data['severe_accident'] = (data['fatality_count'] > 0).astype(int)
    data['severity_score'] = data['fatality_count'] * 2 + data['injury_count']

    num_cols = ['latitude', 'longitude', 'g_v', 'driving_experience', 'alco']
    for col in num_cols:
        data[col] = data[col].fillna(data[col].median())

    cat_cols = ['street_category', 'weather_condition', 'road_condition', 'lighting_condition', 'district', 't_ts',
                'technical_faults']

    for col in ['deficiencies', 'factors']:  # разложение ndu и fact в бинарные признаки
        unique_vals = set()
        data[col].dropna().apply(lambda x: unique_vals.update(x.split(',')))
        for val in unique_vals:
            data[f'{col}_{val}'] = data[col].apply(lambda x: 1 if pd.notnull(x) and val in x else 0)
        data = data.drop(columns=[col])

    data = pd.get_dummies(data, columns=cat_cols, drop_first=False)  # кодирование кат признаков
    scaler = StandardScaler()  # масштабирование числ признаков
    data[num_cols] = scaler.fit_transform(data[num_cols])

    excluded_cols = [
        'emtp_number', 'date_', 'time_', 'accident_type',
        'fatality_count', 'injury_count',
        'severity_score', 'severe_accident']  # искл цел и ненужных признаков

    features = [c for c in data.columns if c not in excluded_cols]
    X = data[features]
    y_type = data['accident_type']
    y_severity_score = data['severity_score']
    y_severe_bin = data['severe_accident']

    X_train_type, X_test_type, y_train_type, y_test_type = train_test_split(
        X, y_type, test_size=0.2, random_state=42)  # тип ДТП

    rf_type = RandomForestClassifier(n_estimators=200, random_state=42)  # RF тип ДТП
    rf_type.fit(X_train_type, y_train_type)
    y_pred_type = rf_type.predict(X_test_type)
    print("Тип ДТП: классификация")
    print(classification_report(y_test_type, y_pred_type, zero_division=0))
    print("Матрица ошибок:")
    print(confusion_matrix(y_test_type, y_pred_type))

    feat_importance_type = pd.DataFrame({
        'feature': X.columns,
        'importance': rf_type.feature_importances_
    }).sort_values(by='importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feat_importance_type.head(20), palette='coolwarm')
    plt.title('Top-20 важных признаков для вида ДТП', fontsize=16)
    plt.xlabel('Важность признака', labelpad=15, fontsize=16)
    plt.ylabel('Признак', labelpad=15, fontsize=16)
    plt.xticks(fontsize=14, rotation=0)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig("rf_type_importance.png")
    plt.show()
    plt.close()

    percentile_99 = y_severity_score.quantile(0.99)  # усечение выбросов тяж ДТП
    y_severity_clipped = np.clip(y_severity_score, None, percentile_99)

    X_train_sev, X_test_sev, y_train_sev, y_test_sev = train_test_split(
        X, y_severity_clipped, test_size=0.2, random_state=42)

    rf_sev = RandomForestRegressor(  # опт параметры
        n_estimators=600,
        max_depth=20,
        min_samples_split=10,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42)
    rf_sev.fit(X_train_sev, y_train_sev, sample_weight=np.sqrt(y_train_sev + 1))  # обучение модели для редких случ

    y_pred_sev = rf_sev.predict(X_test_sev)
    rmse = np.sqrt(mean_squared_error(y_test_sev, y_pred_sev))
    r2 = rf_sev.score(X_test_sev, y_test_sev)

    print("\nТяжесть ДТП: регрессия (оптимизированная)")
    print(f"RMSE: {rmse:.3f}")
    print(f"R^2: {r2:.3f}")

    feat_importance_sev = pd.DataFrame({
        'feature': X.columns,
        'importance': rf_sev.feature_importances_
    }).sort_values(by='importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(
        x='importance',
        y='feature',
        data=feat_importance_sev.head(20),
        palette='coolwarm')
    plt.title('Top-20 важных признаков для тяжести ДТП', fontsize=16)
    plt.xlabel('Важность признака', labelpad=15, fontsize=16)
    plt.ylabel('Признак', labelpad=15, fontsize=16)
    plt.xticks(fontsize=14, rotation=0)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig("rf_sev_importance.png")
    plt.show()
    plt.close()

    X_train_bin, X_test_bin, y_train_bin, y_test_bin = train_test_split(
        X, y_severe_bin, test_size=0.2, random_state=42)  # бин тяжесть ДТП

    ros = RandomOverSampler(random_state=42)
    X_train_bin_res, y_train_bin_res = ros.fit_resample(X_train_bin, y_train_bin)

    print("Размеры выборок до и после апсэмплинга:")
    print(f"До: {np.bincount(y_train_bin)}")
    print(f"После: {np.bincount(y_train_bin_res)}")

    rf_bin = RandomForestClassifier(n_estimators=200, random_state=42)  # обучение модели на сбал данных
    rf_bin.fit(X_train_bin_res, y_train_bin_res)


    y_pred_bin = rf_bin.predict(X_test_bin)
    print("\n=== Тяжесть ДТП (бинарная классификация) ===")
    print(classification_report(y_test_bin, y_pred_bin, zero_division=0))
    print("Матрица ошибок:")
    print(confusion_matrix(y_test_bin, y_pred_bin))

    feat_importance_bin = pd.DataFrame({
        'feature': X.columns,
        'importance': rf_bin.feature_importances_
    }).sort_values(by='importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feat_importance_bin.head(20), palette='coolwarm')
    plt.title('Top-20 важных признаков для бинарной тяжести ДТП', fontsize=16)
    plt.xlabel('Важность признака', labelpad=15, fontsize=16)
    plt.ylabel('Признак', labelpad=15, fontsize=16)
    plt.xticks(fontsize=14, rotation=0)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig("rf_bin_importance.png")
    plt.show()
    plt.close()

    # геогр распределение ДТП
    geo_df = geo_raw[(geo_raw['latitude'] != 0) & (geo_raw['longitude'] != 0)].dropna(subset=['latitude', 'longitude'])
    geo_df['severity_score'] = geo_df['fatality_count'] * 2 + geo_df['injury_count']

    if geo_df.empty:
        print("Нет корректных координат для визуализации.")
    else:
        m = folium.Map(location=[55.03, 82.92], zoom_start=7, tiles='CartoDB positron')
        heat_data = [[row['latitude'], row['longitude'], row['severity_score']] for _, row in
                     geo_df.iterrows()]  # тепловая карта по тяжести ДТП
        HeatMap(heat_data, radius=9, blur=15, max_zoom=1).add_to(m)
        cluster = MarkerCluster().add_to(m)  # кластерные точки с инфой
        for _, row in geo_df.iterrows():
            popup = f"<b>Тип ДТП:</b> {row['accident_type']}<br><b>Тяжесть:</b> {row['severity_score']}"
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=4 + row['severity_score'] * 0.5,
                color='red' if row['severity_score'] > 2 else 'orange',
                fill=True,
                fill_opacity=0.7,
                popup=popup
            ).add_to(cluster)
        m.save("DTP_heatmap.html")
        print("Карта успешно сохранена: DTP_heatmap.html")

        # боксплот видов ДТП
        key_types = [
            "Наезд на велосипедиста",
            "Наезд на пешехода",
            "Столкновение",
            "Съезд с дороги"]

        geo_df['accident_type_grouped'] = geo_df['accident_type'].apply(
            lambda x: x if x in key_types else "Остальные виды")
        ordered_types = key_types + ["Остальные виды"]  # обединение ост видов ДТП
        geo_df['accident_type_grouped'] = pd.Categorical(
            geo_df['accident_type_grouped'],
            categories=ordered_types,
            ordered=True)

        plt.figure(figsize=(10, 6))
        sns.boxplot(
            x='accident_type_grouped',
            y='severity_score',
            data=geo_df,
            palette="coolwarm")
        plt.xticks(rotation=0, ha="center", fontsize=14)
        plt.yticks(fontsize=14)
        plt.title("Распределение тяжести ДТП по видам", fontsize=16)
        plt.xlabel("Вид ДТП", labelpad=15, fontsize=16)
        plt.ylabel("Тяжесть", labelpad=15, fontsize=16)
        plt.tight_layout()
        plt.savefig("severity_by_type.png")
        plt.show()
        plt.close()

        # график районов
        nsr_districts = [
            "Дзержинский", "Железнодорожный", "Заельцовский", "Калининский",
            "Кировский", "Ленинский", "Октябрьский", "Первомайский",
            "Советский", "Центральный"]
        geo_df['region_type'] = np.where(  # создание признака
            geo_df['district'].str.contains('|'.join(nsr_districts), case=False, na=False),
            "Город Новосибирск",
            "Новосибирская область")

        district_stats = geo_df.groupby(["region_type", "district"])["severity_score"].mean().reset_index()

        city_df = district_stats[district_stats['region_type'] == "Город Новосибирск"].sort_values(by='severity_score',
                                                                                                   ascending=False)
        plt.figure(figsize=(10, 6))
        sns.barplot(
            x='district',
            y='severity_score',
            data=city_df,
            palette='Reds_r')
        plt.xticks(rotation=45, ha='right', fontsize=14)
        plt.yticks(fontsize=14)
        plt.title("Средняя тяжесть ДТП по районам г.Новосибирска", fontsize=16)
        plt.ylabel("Средняя тяжесть", labelpad=15, fontsize=16)
        plt.xlabel("Район", labelpad=15, fontsize=16)
        plt.tight_layout()
        plt.savefig("city_districts.png")
        plt.show()
        plt.close()
        print("Средняя тяжесть ДТП по районам г. Новосибирска")
        print(city_df[['district', 'severity_score']])

        region_df = district_stats[district_stats['region_type'] == "Новосибирская область"].sort_values(
            by='severity_score', ascending=False).head(10)
        plt.figure(figsize=(10, 6))
        sns.barplot(
            x='district',
            y='severity_score',
            data=region_df,
            palette='Oranges_r')
        plt.xticks(rotation=45, ha='right', fontsize=14)
        plt.yticks(fontsize=14)
        plt.title("Топ-10 районов Новосибирской области по средней тяжести ДТП", fontsize=16)
        plt.ylabel("Средняя тяжесть", labelpad=15, fontsize=16)
        plt.xlabel("Район", labelpad=15, fontsize=16)
        plt.tight_layout()
        plt.savefig("region_top10.png")
        plt.show()
        plt.close()
        print("\nТоп-10 районов Новосибирской области по средней тяжести ДТП")
        print(region_df[['district', 'severity_score']])

        metrics_dict = {
            "type_classification_report": classification_report(y_test_type, y_pred_type, output_dict=True,
                                                                zero_division=0),
            "type_confusion_matrix": confusion_matrix(y_test_type, y_pred_type),
            "severity_regression": {"RMSE": rmse, "R2": r2},
            "bin_classification_report": classification_report(y_test_bin, y_pred_bin, output_dict=True,
                                                               zero_division=0),
            "bin_confusion_matrix": confusion_matrix(y_test_bin, y_pred_bin)
        }
        report_ready = {
            "rf_type": "rf_type_importance.png",
            "rf_sev": "rf_sev_importance.png",
            "rf_bin": "rf_bin_importance.png",
            "boxplot_type": "severity_by_type.png",
            "city_bar": "city_districts.png",
            "region_bar": "region_top10.png",
            "heatmap": "DTP_heatmap.html"
        }

        return {
            "rf_type": feat_importance_type,
            "rf_sev": feat_importance_sev,
            "rf_bin": feat_importance_bin,
            "metrics": metrics_dict,
            "report_files": report_ready
        }, geo_df


if __name__ == "__main__":
    results, geo_df = main()
    print("Важности признаков (Top-10)")
    for k, df in results.items():
        if isinstance(df, pd.DataFrame):
            print(f"\n{k}:")
            print(df.head(10))

