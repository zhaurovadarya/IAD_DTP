import pickle
import Cor_analysis
import Log_Regr
import Log_Regr_obl
import rand

def main():
    all_results = {}

    print("\n Корреляционный анализ ")
    cor_results = Cor_analysis.main()
    all_results['correlation'] = cor_results

    print("\n Логистическая регрессия (город)")
    log_city_results = Log_Regr.main()
    all_results['log_city'] = log_city_results

    print("\n Логистическая регрессия (область)")
    log_region_results = Log_Regr_obl.main()
    all_results['log_region'] = log_region_results

    print("\n Модели случайного леса")
    rf_results_dict, geo_df = rand.main()
    all_results['random_forest'] = {
        "rf": rf_results_dict,
        "geo": geo_df,
        "report_files": rf_results_dict.get("report_files", {})
    }

    print("\n Все этапы анализа выполнены. Результаты сохранены в analysis_results.pkl")
    return all_results


if __name__ == "__main__":
    main()
