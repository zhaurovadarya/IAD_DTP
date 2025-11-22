import pickle
import Cor_analysis
import Log_Regr
import Log_Regr_obl
import rand
import pr_rules

def main():
    all_results = {}

    print("\n=== 1. Корреляционный анализ ===")
    cor_results = Cor_analysis.main()          # dict {'cor_gor': df, 'cor_obl': df}
    all_results['correlation'] = cor_results

    print("\n=== 2. Логистическая регрессия (город) ===")
    log_city_results = Log_Regr.main()         # dict
    all_results['log_city'] = log_city_results

    print("\n=== 3. Логистическая регрессия (область) ===")
    log_region_results = Log_Regr_obl.main()   # dict
    all_results['log_region'] = log_region_results

    print("\n=== 4. Модели случайного леса ===")
    rf_results_dict, geo_df = rand.main()  # правильно, 2 значения
    all_results['random_forest'] = {
        "rf": rf_results_dict,
        "geo": geo_df,
        "report_files": rf_results_dict.get("report_files", {})
    }

    print("\n=== 5. Формирование факторов для сопоставления ПДД ===")
    df_rules = pr_rules.main()
    all_results['pr_rules'] = df_rules

    # Сохраняем
    with open("analysis_results.pkl", "wb") as f:
        pickle.dump(all_results, f)

    print("\n✅ Все этапы анализа выполнены. Результаты сохранены в analysis_results.pkl")
    return all_results


if __name__ == "__main__":
    main()
