import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sqlalchemy import create_engine

def main():
    engine = create_engine("postgresql+psycopg2://razrab:puk5@localhost:5432/DTP")

    query_gor = """
    WITH hours AS (
        SELECT generate_series(
            TIMESTAMP '2025-01-17 00:00:00',
            TIMESTAMP '2025-01-17 23:00:00',
            INTERVAL '1 hour'
        ) AS datetime_hour
    ),
    accidents AS (
        SELECT 
            DATE_TRUNC('hour', a.date_ + a.time_) AS dt,
            a.emtp_number
        FROM 
            accident a
        JOIN 
            addres ad ON a.address_id = ad.address_id
        WHERE 
            a.date_ = DATE '2025-01-17'
            AND ad.road_value LIKE '%%Местного значения%%'
    ),
    nabludenie AS (
        SELECT 
            DATE_TRUNC('hour', n.date_time) AS dt,
            AVG(n.speed) AS speed,
            SUM(n.proezdy) AS proezdy,
            AVG(n.intensity) AS intensity,
            AVG(n.adjusted_intensity) AS adjusted_intensity,
            AVG(n.load_) AS load_,
            AVG(n.density) AS density,
            AVG(n.average_distance_between_objects) AS average_distance_between_objects,
            AVG(n.time_in_zone) AS time_in_zone
        FROM 
            nabludenie_gor n
        WHERE 
            n.date_time::date = DATE '2025-01-17'
        GROUP BY 
            DATE_TRUNC('hour', n.date_time)
    )
    
    SELECT 
        h.datetime_hour,
        COUNT(DISTINCT a.emtp_number) AS accident_count_gor,
        n.speed,
        n.proezdy,
        n.intensity,
        n.adjusted_intensity,
        n.load_,
        n.density,
        n.average_distance_between_objects,
        n.time_in_zone
    FROM 
        hours h
    LEFT JOIN accidents a ON h.datetime_hour = a.dt
    LEFT JOIN nabludenie n ON h.datetime_hour = n.dt
    GROUP BY 
        h.datetime_hour,
        n.speed,
        n.proezdy,
        n.intensity,
        n.adjusted_intensity,
        n.load_,
        n.density,
        n.average_distance_between_objects,
        n.time_in_zone
    ORDER BY 
        h.datetime_hour;
    """
    df_gor = pd.read_sql(query_gor, engine)
    df_grouped_gor = df_gor.groupby(['datetime_hour']).agg({
        'speed': 'mean',
        'proezdy': 'sum',
        'intensity': 'mean',
        'adjusted_intensity': 'mean',
        'load_': 'mean',
        'density': 'mean',
        'average_distance_between_objects': 'mean',
        'time_in_zone': 'mean',
        'accident_count_gor': 'sum'
    }).reset_index()

    # кор_матрица для городских признаков
    corr_gor = df_grouped_gor[['speed', 'intensity', 'density', 'load_',
                               'average_distance_between_objects', 'accident_count_gor']].corr()

    print("Матрица для городских дорог:")
    print(corr_gor.head())


    rename_map = {
        'average_distance_between_objects': 'avg_dist',
        'accident_count_gor': 'acc_cnt'
    }
    corr_gor_short = corr_gor.rename(index=rename_map, columns=rename_map)
    ax = sns.heatmap(
        corr_gor_short,
        annot=True,
        cmap='coolwarm',
        fmt=".2f",
        cbar_kws={'shrink': 0.8},
        annot_kws={'size': 14}
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha='center', va='top', fontsize=14)
    ax.tick_params(axis='x', pad=15)
    ax.xaxis.tick_top()


    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, ha='right', va='center', fontsize=14)
    ax.tick_params(axis='y', pad=5)

    plt.title("Матрица корреляции для городских признаков", pad=30)
    plt.tight_layout()
    plt.show()

    query_obl = """
    SELECT 
        a.date_ AS accident_date,
        COUNT(a.emtp_number) AS accident_count_obl,
        AVG(no.outward) AS outward,
        AVG(no.return_) AS return
    FROM 
        accident a
    JOIN 
        addres ad ON a.address_id = ad.address_id
    LEFT JOIN 
        nabludenie_obl no ON a.date_ = no.date_
    WHERE 
        ad.road_value IN (
            'Федеральная (дорога федерального значения)', 
            'Региональная или межмуниципальная (дорога регионального или межмуниципального значения)'
        )
    GROUP BY 
        a.date_
    ORDER BY 
        a.date_;
    """
    df_combined = pd.read_sql(query_obl, engine)
    print(df_combined.head())

    corr_obl = df_combined[['outward', 'return', 'accident_count_obl']].corr()
    print("Матрица для областных дорог:")
    print(corr_obl.head())

    sns.heatmap(corr_obl, annot=True, cmap='coolwarm')
    plt.title("Матрица корреляции для областных характеристик")
    plt.show()

    # ГОРОДСКИЕ ДОРОГИ
    mean_acc_gor = df_grouped_gor['accident_count_gor'].mean()
    std_acc_gor = df_grouped_gor['accident_count_gor'].std()
    std_density = df_grouped_gor['density'].std()
    std_dist = df_grouped_gor['average_distance_between_objects'].std()

    # коэффициенты корреляции
    r_density = corr_gor.loc['density', 'accident_count_gor']
    r_dist = corr_gor.loc['average_distance_between_objects', 'accident_count_gor']

    # изменения признаков
    delta_density = -7.5
    delta_dist = -2.0

    # расчет изменения ДТП
    delta_acc_density = r_density * (std_acc_gor / std_density) * delta_density
    delta_acc_dist = r_dist * (std_acc_gor / std_dist) * delta_dist

    percent_density = (delta_acc_density / mean_acc_gor) * 100
    percent_dist = (delta_acc_dist / mean_acc_gor) * 100

    print("\nГОРОДСКИЕ ДОРОГИ ")
    print(f"r_density = {r_density:.3f}, r_dist = {r_dist:.3f}")
    print(f"Среднее ДТП: {mean_acc_gor:.2f}, std ДТП: {std_acc_gor:.2f}")
    print(f"Δплотности: {delta_density:.2f} → ΔДТП: {delta_acc_density:+.2f} шт. (~{percent_density:.1f}%)")
    print(f"Δдистанции: {delta_dist:+.1f} м → ΔДТП: {delta_acc_dist:+.2f} шт. (~{percent_dist:.1f}%)")

    # ОБЛАСТНЫЕ ДОРОГИ
    mean_acc_obl = df_combined['accident_count_obl'].mean()
    std_acc_obl = df_combined['accident_count_obl'].std()
    std_outward = df_combined['outward'].std()
    std_return = df_combined['return'].std()

    # коэффициенты корреляции
    r_outward = corr_obl.loc['outward', 'accident_count_obl']
    r_return = corr_obl.loc['return', 'accident_count_obl']

    # изменения признаков
    delta_outward = 1000
    delta_return = 500

    # расчет изменения ДТП
    delta_acc_outward = r_outward * (std_acc_obl / std_outward) * delta_outward
    delta_acc_return = r_return * (std_acc_obl / std_return) * delta_return

    percent_outward = (delta_acc_outward / mean_acc_obl) * 100
    percent_return = (delta_acc_return / mean_acc_obl) * 100

    print("\n ОБЛАСТНЫЕ ДОРОГИ ")
    print(f"r_outward = {r_outward:.3f}, r_return = {r_return:.3f}")
    print(f"Среднее ДТП: {mean_acc_obl:.2f}, std ДТП: {std_acc_obl:.2f}")
    print(f"Δтрафика наружу: +{delta_outward} авто → ΔДТП: {delta_acc_outward:+.2f} шт. (~{percent_outward:.1f}%)")
    print(f"Δтрафика внутрь: +{delta_return} авто → ΔДТП: {delta_acc_return:+.2f} шт. (~{percent_return:.1f}%)")
    corr_gor = df_grouped_gor[['speed', 'intensity', 'density', 'load_',
                               'average_distance_between_objects', 'accident_count_gor']].corr()
    corr_obl = df_combined[['outward', 'return', 'accident_count_obl']].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        corr_gor.rename(index=rename_map, columns=rename_map),
        annot=True, cmap="coolwarm", fmt=".2f",
        cbar_kws={'shrink': 0.8},
        annot_kws={'size': 14}
    )
    plt.title("Матрица корреляции для городских признаков", pad=30)
    plt.tight_layout()
    gor_plot_path = "corr_gor.png"
    plt.savefig(gor_plot_path)
    plt.close()

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_obl, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Матрица корреляции для областных характеристик")
    plt.tight_layout()
    obl_plot_path = "corr_obl.png"
    plt.savefig(obl_plot_path)
    plt.close()

    return {
        "cor_gor": corr_gor,
        "cor_obl": corr_obl,
        "gor_plot": gor_plot_path,
        "obl_plot": obl_plot_path,
        "gor_interpretation": {
            "r_density": float(r_density),
            "r_dist": float(r_dist),
            "mean_acc": float(mean_acc_gor),
            "std_acc": float(std_acc_gor),
            "delta_acc_density": float(delta_acc_density),
            "delta_acc_dist": float(delta_acc_dist),
            "percent_density": float(percent_density),
            "percent_dist": float(percent_dist),
        },
        "obl_interpretation": {
            "r_outward": float(r_outward),
            "r_return": float(r_return),
            "mean_acc": float(mean_acc_obl),
            "std_acc": float(std_acc_obl),
            "delta_acc_outward": float(delta_acc_outward),
            "delta_acc_return": float(delta_acc_return),
            "percent_outward": float(percent_outward),
            "percent_return": float(percent_return),
        }
    }


if __name__ == "__main__":
    main()