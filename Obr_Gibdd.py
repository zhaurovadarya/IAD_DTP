import json
import pandas as pd
from sqlalchemy import create_engine

engine = create_engine('postgresql+psycopg2://razrab:puk5@localhost:5432/DTP')
def load_csv_to_db(
        csv_path,
        table_name,
        engine,
        id_prefix=None,
        id_column=None,
        rename_columns=None,
        fillna_values=None,
        clean_strings=True
):
    df = pd.read_csv(csv_path)
    if rename_columns:
        df = df.rename(columns=rename_columns)
    if fillna_values:
        df.fillna(fillna_values, inplace=True)
    if clean_strings:
        df = df.apply(lambda col: col.str.strip() if col.dtype == "object" else col)

    # авто-генерация ID без пересечения с существующими
    if id_prefix and id_column:
        try:
            existing_ids = pd.read_sql(f"SELECT {id_column} FROM {table_name}", con=engine)[id_column]
            if existing_ids.empty:
                start = 1
            else:
                numeric_ids = existing_ids.str.extract(r'(\d+)').astype(float).fillna(0)[0]
                start = int(numeric_ids.max()) + 1
        except:
            start = 1
        df[id_column] = [f"{id_prefix}{i}" for i in range(start, start + len(df))]

    df.to_sql(table_name, con=engine, index=False, if_exists='append', method='multi')
    print(f"Данные из {csv_path} успешно загружены в таблицу {table_name}")
    return df
df_participants = load_csv_to_db(
    csv_path=r"C:\Users\Дарья\Desktop\BigData\!Анализ на ПАЙТОНЕ\курсач\IAD-main\IAD-main\IAD\parti_short_25.csv",
    table_name="participants",
    engine=engine
)

df_addresses = load_csv_to_db(
    csv_path=r"C:\Users\Дарья\Desktop\BigData\!Анализ на ПАЙТОНЕ\курсач\IAD-main\IAD-main\IAD\addres_short.csv",
    table_name="addres",
    engine=engine,
    id_prefix="A",
    id_column="address_id",
    rename_columns={
        'District': 'district',
        'dor_z': 'road_value',
        'k_ul': 'street_category'
    },
    fillna_values={
        'district': '0',
        'road_value': '0',
        'street_category': '0',
        'n_p': '0',
        'street': '0',
        'house': '0',
        'dor': '0',
        'km': 0
    }
)

df_influence_csv = load_csv_to_db(
    csv_path=r"C:\Users\Дарья\Desktop\BigData\!Анализ на ПАЙТОНЕ\курсач\IAD-main\IAD-main\IAD\influence_short.csv",
    table_name="accident_influence",
    engine=engine,
    id_prefix="Inf",
    id_column="influence_id"
)

df_dtp = pd.read_csv(r"C:\Users\Дарья\Desktop\BigData\!Анализ на ПАЙТОНЕ\курсач\IAD-main\IAD-main\IAD\dtp_short.csv").fillna('0')
df_dtp = df_dtp.apply(lambda col: col.str.strip() if col.dtype == "object" else col)

df_addr_db = pd.read_sql("SELECT * FROM addres", con=engine)
df_infl_db = pd.read_sql("SELECT * FROM accident_influence", con=engine)

df_dtp = df_dtp.merge(df_addr_db, how='left',
                      on=['district', 'road_value', 'street_category', 'n_p', 'street', 'house', 'dor', 'km'])
df_dtp = df_dtp.merge(df_infl_db, how='left',
                      on=['weather_condition', 'road_condition', 'lighting_condition', 'traffic_rules'])

final_df = df_dtp[['emtp_number', 'address_id', 'influence_id', 'date_', 'latitude', 'longitude', 'time_',
                    'accident_type', 'vehicle_count', 'participant_count', 'fatality_count', 'injury_count']]
final_df = final_df.drop_duplicates(subset='emtp_number', keep='first')

final_df.to_sql('accident', con=engine, index=False, if_exists='append', method='multi')
print("Данные успешно загружены в таблицу 'accident'")

df_vehicles = load_csv_to_db(
    csv_path=r"C:\Users\Дарья\Desktop\BigData\!Анализ на ПАЙТОНЕ\курсач\IAD-main\IAD-main\IAD\vehicles_short.csv",
    table_name="vehicles",
    engine=engine,
    fillna_values={'r_rul': '0', 'model': '0', 'ownership_form': '0', 'technical_faults': '0'}
)

df_offenses = load_csv_to_db(
    csv_path=r"C:\Users\Дарья\Desktop\BigData\!Анализ на ПАЙТОНЕ\курсач\IAD-main\IAD-main\IAD\Associated_Offenses_short.csv",
    table_name="associated_offenses",
    engine=engine,
    fillna_values=''
)

df_infl_25 = load_csv_to_db(
    csv_path=r"C:\Users\Дарья\Desktop\BigData\!Анализ на ПАЙТОНЕ\курсач\IAD-main\IAD-main\IAD\influence_short_25.csv",
    table_name="accident_influence",
    engine=engine,
    id_prefix="Inf",
    id_column="influence_id"
)

df_influence = pd.read_sql("SELECT * FROM accident_influence", con=engine)
df_def = pd.read_sql("SELECT * FROM deficiencies", con=engine)
df_fac = pd.read_sql("SELECT * FROM factors", con=engine)

with open(r"C:\Users\Дарья\Desktop\BigData\!Анализ на ПАЙТОНЕ\курсач\IAD-main\IAD-main\IAD\parsed_2025.json", encoding="utf-8") as f:
    parsed_data = json.load(f)

if isinstance(parsed_data, dict):
    data = json.loads(parsed_data['data'])
    cards = data.get("cards", [])

    inf_def_rows, inf_fac_rows = [], []
    def_set, fac_set = set(), set()
    existing_def_ids = pd.read_sql("SELECT influence_deficiency_id FROM influence_deficiency", con=engine)
    id_counter_def = int(existing_def_ids.influence_deficiency_id.str.extract(r'(\d+)').max() or 519) + 1

    existing_fac_ids = pd.read_sql("SELECT influence_factor_id FROM influence_factor", con=engine)
    id_counter_fac = int(existing_fac_ids.influence_factor_id.str.extract(r'(\d+)').max() or 189) + 1

    for entry in cards:
        info = entry.get("infoDtp", {})
        s_pog = info.get("s_pog")
        weather = s_pog[0] if isinstance(s_pog, list) and s_pog else s_pog if isinstance(s_pog, str) else ""
        influence = {
            "weather_condition": weather,
            "road_condition": info.get("s_pch", ""),
            "lighting_condition": info.get("osv", ""),
            "traffic_rules": info.get("change_org_motion", "")
        }

        influence_row = df_influence[
            (df_influence["weather_condition"] == influence["weather_condition"]) &
            (df_influence["road_condition"] == influence["road_condition"]) &
            (df_influence["lighting_condition"] == influence["lighting_condition"]) &
            (df_influence["traffic_rules"] == influence["traffic_rules"])
        ]
        if influence_row.empty:
            continue
        influence_id = influence_row.iloc[0]["influence_id"]

        for ndu in info.get("ndu", []):
            def_row = df_def[df_def["ndu"] == ndu]
            if def_row.empty:
                continue
            deficiency_id = def_row.iloc[0]["deficiency_id"]
            key = (influence_id, deficiency_id)
            if key not in def_set:
                inf_def_rows.append({
                    "influence_deficiency_id": f"ID{id_counter_def}",
                    "influence_id": influence_id,
                    "deficiency_id": deficiency_id
                })
                def_set.add(key)
                id_counter_def += 1

        for fac in info.get("factor", []):
            fac_row = df_fac[df_fac["factor"] == fac]
            if fac_row.empty:
                continue
            factor_id = fac_row.iloc[0]["factor_id"]
            key = (influence_id, factor_id)
            if key not in fac_set:
                inf_fac_rows.append({
                    "influence_factor_id": f"IF{id_counter_fac}",
                    "influence_id": influence_id,
                    "factor_id": factor_id
                })
                fac_set.add(key)
                id_counter_fac += 1

    if inf_def_rows:
        pd.DataFrame(inf_def_rows).to_sql("influence_deficiency", con=engine, index=False, if_exists="append", method="multi")
    if inf_fac_rows:
        pd.DataFrame(inf_fac_rows).to_sql("influence_factor", con=engine, index=False, if_exists="append", method="multi")
df_addr = pd.read_sql("SELECT address_id, n_p, street, house FROM addres", con=engine)
df_sdor = pd.read_sql("SELECT uds_object_id, sdor FROM objects_accident_site", con=engine)
df_obj = pd.read_sql("SELECT uds_object_nearby_id, obj_dtp FROM objects_near_accident_site", con=engine)

df_sdor["sdor_norm"] = df_sdor["sdor"].str.strip().str.lower()
df_obj["obj_dtp_norm"] = df_obj["obj_dtp"].str.strip().str.lower()

with open(r"C:\Users\Дарья\Desktop\BigData\!Анализ на ПАЙТОНЕ\курсач\IAD-main\IAD-main\IAD\parsed_2024.json", encoding="utf-8") as f:
    parsed_data_2024 = json.load(f)

addr_obj_rows, addr_near_rows = [], []
seen_obj, seen_near = set(), set()

for entry in parsed_data_2024:
    for card in entry.get("cards", []):
        info = card.get("infoDtp", {})
        n_p = info.get("n_p", "")
        street = info.get("street", "")
        house = str(info.get("house", "0"))

        address_row = df_addr[
            (df_addr["n_p"] == n_p) &
            (df_addr["street"] == street) &
            (df_addr["house"] == house)
        ]
        if address_row.empty:
            continue
        address_id = address_row.iloc[0]["address_id"]

        for val in info.get("sdor", []):
            val_norm = val.strip().lower()
            match = df_sdor[df_sdor["sdor_norm"] == val_norm]
            if not match.empty:
                uds_object_id = match.iloc[0]["uds_object_id"]
                key = (address_id, uds_object_id)
                if key not in seen_obj:
                    addr_obj_rows.append({"address_id": address_id, "uds_object_id": uds_object_id})
                    seen_obj.add(key)

        for val in info.get("OBJ_DTP", []):
            val_norm = val.strip().lower()
            match = df_obj[df_obj["obj_dtp_norm"] == val_norm]
            if not match.empty:
                uds_object_nearby_id = match.iloc[0]["uds_object_nearby_id"]
                key = (address_id, uds_object_nearby_id)
                if key not in seen_near:
                    addr_near_rows.append({"address_id": address_id, "uds_object_nearby_id": uds_object_nearby_id})
                    seen_near.add(key)

if addr_obj_rows:
    pd.DataFrame(addr_obj_rows).to_sql("address_objects_accident_site", con=engine, index=False, if_exists="append", method="multi")
if addr_near_rows:
    pd.DataFrame(addr_near_rows).to_sql("address_objects_near_accident_site", con=engine, index=False, if_exists="append", method="multi")

print("Объединенная загрузка CSV и JSON с авто-ID завершена")

df_acc = pd.read_sql("SELECT emtp_number, address_id FROM accident", con=engine)
df_addr = pd.read_sql("SELECT address_id, dor, street FROM addres", con=engine)
df_station = pd.read_sql("SELECT id_station, uchastock_name_obl FROM station", con=engine)
df_uch = pd.read_sql("SELECT id_uchastock, name_uchastock FROM uchastock", con=engine)

# связь с областными участками (address.dor - station.uchastock_name_obl)
df_obl = df_addr.merge(df_station, left_on='dor', right_on='uchastock_name_obl', how='inner')
df_obl = df_obl.merge(df_acc, on='address_id', how='inner')

# авто-генерация ID для областных связей
try:
    existing_ids_obl = pd.read_sql("SELECT id_link_obl FROM link_dtp_obl_uchastok", con=engine)
    start_obl = int(existing_ids_obl.id_link_obl.str.extract(r'(\d+)').max()) + 1
except:
    start_obl = 1
df_obl['id_link_obl'] = ['L_obl_' + str(i) for i in range(start_obl, start_obl + len(df_obl))]

# связь с городскими участками (address.street - uchastock.name_uchastock)
df_gor = df_addr.merge(df_uch, left_on='street', right_on='name_uchastock', how='inner')
df_gor = df_gor.merge(df_acc, on='address_id', how='inner')

# авто-генерация ID для городских связей
try:
    existing_ids_gor = pd.read_sql("SELECT id_link_gor FROM link_dtp_gor_uchastok", con=engine)
    start_gor = int(existing_ids_gor.id_link_gor.str.extract(r'(\d+)').max()) + 1
except:
    start_gor = 1
df_gor['id_link_gor'] = ['L_gor_' + str(i) for i in range(start_gor, start_gor + len(df_gor))]

# финальные таблицы
df_gor_final = df_gor[['id_link_gor', 'emtp_number', 'address_id', 'id_uchastock']]
df_gor_final.to_sql("link_dtp_gor_uchastok", con=engine, index=False, if_exists='append', method='multi')
print("Загружено в link_dtp_gor_uchastok")

print("Размеры таблиц:")
print("addres:", df_addr.shape)
print("uchastock:", df_uch.shape)
print("городские связи:", df_gor.shape)
