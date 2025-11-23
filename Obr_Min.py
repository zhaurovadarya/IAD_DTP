import os
import pandas as pd
import re
import psycopg2
from sqlalchemy import create_engine

base_dir = r"C:\Users\Дарья\Desktop\BigData\!Анализ на ПАЙТОНЕ\курсач\IAD-main\IAD-main\IAD\Обл_2024"
all_dataframes = []

for root, dirs, files in os.walk(base_dir):
    for file in files:
        if file.endswith('.xls') or file.endswith('.xlsx'):
            file_path = os.path.join(root, file)
            try:
                df = pd.read_excel(file_path)
                df['Source_File'] = file
                all_dataframes.append(df)
            except Exception as e:
                print(f"Ошибка при чтении файла {file_path}: {e}")

# все датафреймы в один
if all_dataframes:
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    print(f"Успешно объединено файлов: {len(all_dataframes)}")
    save_path = r"C:\Users\Дарья\Desktop\BigData\!Анализ на ПАЙТОНЕ\курсач\IAD-main\IAD-main\IAD\Together_obl.xlsx"
    combined_df.to_excel(save_path, index=False)
    print(f"Файл успешно сохранен по пути: {save_path}")
else:
    print("Файлы не найдены или ошибка при чтении.")

#ЗАПОЛНЕНИЕ таблиц Nabludenie_gor
file_path = 'Together_gor.xlsx'
df = pd.read_excel(file_path)
conn = psycopg2.connect(
    dbname="DTP",
    user="razrab",
    password="puk5",
    host="localhost",
    port="5432"
)
conn.autocommit = True
cur = conn.cursor()
cur.execute("DELETE FROM Intensity_type_Vehicles;")
cur.execute("DELETE FROM Nabludenie_gor;")
type_columns = [col for col in df.columns if col.startswith('Type_')]
global_intensity_counter = 1
for idx, row in df.iterrows():
    id_uchastock = row['id_uchastock']
    id_polosa = row['ID_polosa']
    insert_nabludenie_query = """
        INSERT INTO Nabludenie_gor (
            ID_Uchastock, ID_polosa, Date_time, Speed, Proezdy, Intensity, Adjusted_intensity,
            Load_, Density, Average_distance_between_objects, Time_in_zone
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING ID_Nabludenie_gor;
    """
    cur.execute(insert_nabludenie_query, (
        id_uchastock,
        id_polosa,
        row['Date_time'],
        row['Speed'],
        row['Proezdy'],
        row['Intensity'],
        row['Adjusted_intensity'],
        row['Load_'],
        row['Density'],
        row['Average_distance_between_objects'],
        row['Time_in_zone']
    ))
    id_nabludenie_gor = cur.fetchone()[0]
    for type_col in type_columns:
        intensity_value = row[type_col]
        if pd.notnull(intensity_value) and intensity_value != 0:
            insert_intensity_query = """
                INSERT INTO Intensity_type_Vehicles (ID_Intensity, ID_Type_Vehicles, ID_Nabludenie_gor, Intensity_per_hour)
                VALUES (%s, %s, %s, %s);
            """
            id_intensity = f"I_{global_intensity_counter}"  # Глобальный счётчик ID
            cur.execute(insert_intensity_query, (
                id_intensity,
                type_col,
                id_nabludenie_gor,
                intensity_value
            ))
            global_intensity_counter += 1

print("Загрузка данных завершена успешно!")
cur.close()
conn.close()
#ЗАПОЛНЕНИЕ таблиц Nabludenie_obl
engine = create_engine('postgresql+psycopg2://razrab:puk5@localhost:5432/DTP')
file_path = r"C:\Users\Дарья\Desktop\Полетайкин\!Анализ на ПАЙТОНЕ\курсач\IAD-main\IAD-main\IAD\Together_obl_cleaned.xlsx"
df = pd.read_excel(file_path)
df = df.rename(columns={
    'Type_3': 'type_3_outward', 'Type_3.1': 'type_3_return',
    'Type_7': 'type_7_outward', 'Type_7.1': 'type_7_return',
    'Type_8': 'type_8_outward', 'Type_8.1': 'type_8_return',
    'Type_6': 'type_6_outward', 'Type_6.1': 'type_6_return',
    'Type_9': 'type_9_outward', 'Type_9.1': 'type_9_return',
    'Type_10': 'type_10_outward', 'Type_10.1': 'type_10_return',
    'Отчет по трафику': 'date_'
})

df = df[pd.to_datetime(df['date_'], errors='coerce').notna()]
df['date_'] = pd.to_datetime(df['date_']).dt.date
vehicle_types = ['type_3', 'type_7', 'type_8', 'type_6', 'type_9', 'type_10']

observations = []
for _, row in df.iterrows():
    for vt in vehicle_types:
        outward = pd.to_numeric(row.get(f"{vt}_outward", 0), errors='coerce') or 0
        return_ = pd.to_numeric(row.get(f"{vt}_return", 0), errors='coerce') or 0
        observations.append({
            'id_station': int(row['id_station']),
            'id_type_vehicles': vt.title(),
            'date_': row['date_'],
            'outward': int(outward),
            'return_': int(return_)
        })
if observations:
    obs_df = pd.DataFrame(observations)
    obs_df.to_sql('nabludenie_obl', con=engine, index=False, if_exists='append', method='multi')
    print(f"Загружено {len(obs_df)} строк в 'nabludenie_obl'")
else:
    print("Нет данных для загрузки.")
