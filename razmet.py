import pandas as pd
import re
import pymorphy2
from tqdm import tqdm

morph = pymorphy2.MorphAnalyzer()
def lemmatize_text(text):
    words = re.findall(r"[а-яА-Яa-zA-ZёЁ]+", str(text).lower())
    lemmas = [morph.parse(w)[0].normal_form for w in words]
    return lemmas  # возвращаем список лемм

df = pd.read_csv("df_rules.csv", low_memory=False)
pdd = pd.read_csv("PDD_rules.csv")  # колонки: id, text

print(f"df_rules.csv: {df.shape[1]} признаков, {df.shape[0]} строк")
print(f"PDD_rules.csv: {pdd.shape[0]} пунктов ПДД")
df_cols_lemmas = {col: lemmatize_text(col) for col in df.columns} # Лемматизация признаков
themes = {
    "Погодные условия": [
        "Ясно", "Снегопад", "Пасмурно", "Дождь", "Метель", "Туман" ],
    "Освещение": [
        "В темное время суток, освещение включено",
        "В темное время суток, освещение отсутствует",
        "В темное время суток, освещение не включено",
        "Сумерки",
        "Светлое время суток"],
    "Состояние дороги": [
        "Заснеженное", "Сухое", "Мокрое",
        "Обработанное противогололедными материалами",
        "Со снежным накатом", "Гололедица",
        "Свежеуложенная поверхностная обработка",
        "Загрязненное", "Пыльное"],
    "Тип ТС": [
        "В-класс (малый) до 3,9 м",
        "С-класс (малый средний, компактный) до 4,3 м",
        "D-класс (средний) до 4,6 м",
        "Е-класс (высший средний, бизнес-класс) до 4,9 м",
        "Прочие легковые автомобили",
        "Одноэтажные длиной от 5 до 8 м",
        "Одноэтажные длиной от 8 до 12 м",
        "Одноэтажные длиной не более 5 м",
        "Мопеды с двигателем внутреннего сгорания менее 50 см. куб.",
        "Мопеды с двигателем внутреннего сгорания более 50 см. куб.",
        "Мопеды с электродвигателем менее 4 кВт",
        "Минивэны и универсалы повышенной вместимости",
        "Троллейбусы",
        "Прочая спецтехника",
        "Седельные тягачи",
        "Рефрижераторы",
        "Самосвалы",
        "Тракторы",
        "Экскаваторы",
        "Фронтальные погрузчики",
        "Цистерны",
        "Квадроциклы",
        "Автоэвакуаторы",
        "Автогрейдеры",
        "Прицепы прочие",
        "Прицепы к легковым автомобилям",
        "Прицепы тракторные",
        "Полуприцепы-фургоны"]}

print("Лемматизация текста ПДД...")
tqdm.pandas()
pdd['lemma_text'] = pdd['text'].progress_apply(lemmatize_text)

merged = [] # сопоставление пдд и признаков
for _, row in tqdm(pdd.iterrows(), total=pdd.shape[0], desc="Сопоставление"):
    lemma_text = row['lemma_text']
    related_factors = []
    related_themes = set()

    for col, col_lemmas in df_cols_lemmas.items():
        if any(l in lemma_text for l in col_lemmas):
            related_factors.append(col)
            for theme, keywords in themes.items():
                if any(k.lower() in col.lower() for k in keywords):
                    related_themes.add(theme)

    merged.append({
        "pdd_id": row.get('id', None),
        "pdd_text": row['text'],
        "related_factors": ", ".join(related_factors) if related_factors else "",
        "themes": ", ".join(related_themes) if related_themes else "Другое"})
merged_df = pd.DataFrame(merged)
merged_df.to_csv("razmetPF.csv", index=False, encoding="utf-8-sig")
print(f"\nИтоговая таблица сохранена: razmetPF.csv")
print(f"Сопоставлено пунктов ПДД: {(merged_df['related_factors'] != '').sum()} из {len(merged_df)}")
print(f"Процент охвата: {(merged_df['related_factors'] != '').sum() / len(merged_df) * 100:.1f}%")


