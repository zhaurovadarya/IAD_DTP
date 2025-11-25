#!/usr/bin/python
# -*- coding: UTF-8 -*-
import requests
import json
from datetime import datetime
import codecs
import os
import re
import sys
import argparse
log_filename = "parselog.log"
def create_log():
    with open(log_filename, 'w') as f:
        pass
def write_log(text):
    timestamp = datetime.strftime(datetime.now(), "%Y-%m-%d %H:%M:%S")
    with open(log_filename, 'a') as f:
        f.write("{} {}".format(timestamp, text))
        f.write("\n")
def getLatestDate():
    year = datetime.now().year
    if datetime.now().month > 2:
        last_month = datetime.now().month - 1
    else:
        year -= 1
        last_month = 12
    return {"month":last_month, "year":year}
def getRusFedData():
    latest_m_y = getLatestDate()
    rf_dict = {"maptype":1,"region":"877","date":"[\"MONTHS:{0}.{1}\"]".format(latest_m_y["month"], latest_m_y["year"]),"pok":"1"}
    r = requests.post("http://stat.gibdd.ru/", json=rf_dict)
    if (r.status_code != 200):
        log_text = u"Не удалось получить данные по регионам РФ"
        print(log_text)
        write_log(log_text)
        return None
    else:
        log_text = u"Получены данные по регионам РФ"
        print(log_text)
        write_log(log_text)
        return r.content


# пары код ОКАТО + название региона
def getRegionsInfo():
    content = getRusFedData()
    if content == None:
        return None
    else:
        regions = []
        d = (json.loads(content))
        regions_dict = json.loads(json.loads(d["metabase"])[0]["maps"])
        for rd in regions_dict:
            regions.append({"id": rd["id"], "name": rd["name"]})
        return regions
# шаг 2) получаем ОКАТО-коды муниципальных образований для всех регионов
#по умолчанию берем самые свежие данные, за месяц перед текущим
def getRegionData(region_id, region_name):
    latest_m_y = getLatestDate()
    region_dict = {"maptype":1,"date":"[\"MONTHS:{0}.{1}\"]".format(latest_m_y["month"], latest_m_y["year"]), "pok": "1"}
    region_dict["region"] = region_id  # region_id: string
    r = requests.post("http://stat.gibdd.ru/map/getMainMapData", json=region_dict)
    if (r.status_code != 200):
        log_text = u"Не удалось получить статистику по региону {0} {1}".format(region_id, region_name)
        print(log_text)
        write_log(log_text)
        return None
    else:
        log_text = u"Получена статистика по региону {0} {1}".format(region_id, region_name)
        print(log_text)
        write_log(log_text)
        return r.content
# пары код ОКАТО + название муниципального образования для всех регионов
def getDistrictsInfo(region_id, region_name):
    content = getRegionData(region_id, region_name)
    if content == None:
        return None
    else:
        d = (json.loads(content))
        district_dict = json.loads(json.loads(d["metabase"])[0]["maps"])
        districts = []
        for dd in district_dict:
            districts.append({"id": dd["id"], "name": dd["name"]})
        return json.dumps(districts).encode('utf8').decode('unicode-escape')
# сохраняем справочник ОКАТО-кодов и названий регионов и муниципалитетов
def saveCodeDictionary(filename):
    region_codes = getRegionsInfo()
    for region in region_codes:
        region["districts"] = getDistrictsInfo(region["id"], region["name"])
    with codecs.open(filename, "w", encoding="utf-8") as f:
        json.dump(region_codes, f, ensure_ascii=False)
# шаг 3) получаем карточки ДТП по заданному региону за указанный период
#st и en - номер первой и последней карточки, т.к. на ресурсе - постраничный перебор данных
def getDTPData(region_id, region_name, district_id, district_name, months, year):
    cards_dict = {
        "data": {
            "date": ["MONTHS:1.2017"],
            "ParReg": "71100",
            "order": {"type": "1", "fieldName": "dat"},
            "reg": "71118",
            "ind": "1",
            "st": "1",
            "en": "16"
        }
    }

    cards_dict["data"]["ParReg"] = region_id
    cards_dict["data"]["reg"] = district_id
    months_list = ["MONTHS:" + str(month) + "." + str(year) for month in months]
    cards_dict["data"]["date"] = months_list

    start = 1
    increment = 50
    json_data = None

    # логируем начало обработки
    if len(months) == 1:
        log_text = u"Обрабатываются данные для {0} ({1}) за {2}.{3}".format(
            region_name, district_name, months[0], year
        )
    else:
        log_text = u"Обрабатываются данные для {0} ({1}) за {2}-{3}.{4}".format(
            region_name, district_name, months[0], months[-1], year
        )
    print(log_text)
    write_log(log_text)

    while True:
        cards_dict["data"]["st"] = str(start)
        cards_dict["data"]["en"] = str(start + increment - 1)

        cards_dict_json = {
            "data": json.dumps(cards_dict["data"], separators=(',', ':')).encode('utf8').decode('unicode-escape')
        }

        r = requests.post("http://stat.gibdd.ru/map/getDTPCardData", json=cards_dict_json)

        if r.status_code == 200:
            try:
                cards = json.loads(json.loads(r.content)["data"])["tab"]
            except:
                log_text = u"Отсутствуют данные для {0} ({1}) за {2}-{3}.{4}".format(
                    region_name, district_name, months[0], months[-1], year
                )
                print(log_text)
                write_log(log_text)
                break

            if len(cards) > 0:
                if json_data is None:
                    json_data = cards
                else:
                    json_data += cards

            if len(cards) == increment:
                start += increment
            else:
                break
        else:
            if "Unexpected character (',' (code 44))" in r.text:
                break
            else:
                log_text = u"Отсутствуют данные для {0} ({1}) за {2}-{3}.{4}".format(
                    region_name, district_name, months[0], months[-1], year
                )
                print(log_text)
                write_log(log_text)
                break

    return json_data

# шаг 4) сохраняем статистику ДТП. один файл = один регион за один год

def getDTPInfo(data_root, year, months, regions, region_id="0"):
    data_dir = os.path.join(data_root, str(year))
    regions_downloaded = []
    if os.path.exists(data_dir):
        files = [x for x in os.listdir(data_dir) if x.endswith(".json")]
        for file in files:
            result = re.match("([0-9]+)([^0-9]+)(.*)", file)
            regions_downloaded.append(result.group(2).strip())
    for region in regions:
        if region["name"] != "Новосибирская область":
            continue
        if region["name"] in regions_downloaded:
            log_text = u"Статистика по региону {} уже загружена".format(region["name"])
            print(log_text)
            write_log(log_text)
            continue
        dtp_dict = {"data": {}}
        dtp_dict["data"]["year"] = str(year)
        dtp_dict["data"]["region_code"] = region["id"]
        dtp_dict["data"]["region_name"] = region["name"]
        dtp_dict["data"]["month_first"] = months[0]
        dtp_dict["data"]["month_last"] = months[-1]
        dtp_dict["data"]["cards"] = []
        # муниципальные образования в регионе
        districts = json.loads(region["districts"])
        for district in districts:
            # получение карточек ДТП
            log_text = u"Обрабатываются данные для {0} ({1}) за {2}-{3}.{4}".\
                  format(region["name"], district["name"], months[0], months[-1], year)
            print(log_text)
            write_log(log_text)
            cards = getDTPData(region["id"], region["name"], district["id"], district["name"], months, year)
            if cards == None:
                continue
            log_text = u"{0} ДТП для {1} ({2}) за {3}-{4}.{5}".format(len(cards), region["name"], district["name"],
                                                                      months[0], months[len(months) - 1], year)
            print(log_text)
            write_log(log_text)
            dtp_dict["data"]["cards"] += cards
        dtp_dict_json = {}
        dtp_dict_json["data"] = json.dumps(dtp_dict["data"]).encode('utf8').decode('unicode-escape')
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        filename = os.path.join(data_dir, "{} {} {}-{}.{}.json".format(region["id"], region["name"], months[0], months[len(months) - 1], year))
        with codecs.open(filename, "w", encoding="utf-8") as f:
            json.dump(dtp_dict_json, f, ensure_ascii=False, separators=(',', ':'))
            log_text = u"Сохранены данные для {} за {}-{}.{}".format(region["name"], months[0], months[len(months) - 1], year)
            print(log_text)
            write_log(log_text)
        if region["id"] == region_id:
            break

def createParser():
    parser = argparse.ArgumentParser(description="GibddStatParser.py [--year] [--month] [--regcode] [--dir] [--updatecodes] [--help]")
    parser.add_argument('--year', type=str,
        help = u'год, за который скачивается статистика. пример: --year 2017')
    parser.add_argument('--month', type=str,
        help = u'месяц, за который скачивается статистика. пример: --month 1. не указан - скачиваются все')
    parser.add_argument('--regcode', default='0', type=str,
        help = u'ОКАТО-код региона (см. в regions.json). пример для Москвы: --regcode 45. не указан - скачиваются все')
    parser.add_argument('--dir', default='dtpdata', type=str,
        help = u'каталог для сохранения карточек ДТП. по умолчанию dtpdata')
    parser.add_argument('--updatecodes', default='n', help = u'обновить справочник регионов. пример: --updatecodes y')
    return parser
def getParamSplitted(param, command_name):
    splitted_list = []
    splitted = param.split("-")
    try:
        splitted_list.append(int(splitted[0]))
        if len(splitted) == 2:
            splitted_list.append(int(splitted[1]))
    except:
        log_text = u"Неверное значение параметра {}".format(command_name)
        print(log_text)
        write_log(log_text)
    return splitted_list
def main():
    parser = createParser()
    namespace = parser.parse_args(sys.argv[1:])
    data_root = "C:\\Users\\Дарья\\Desktop\\BigData\\!Анализ на ПАЙТОНЕ\\курсач\\IAD-main\\IAD-main\\IAD"
    if not os.path.exists(data_root):
        os.makedirs(data_root)
    if not os.path.exists(log_filename):
        create_log()
    if len(namespace.updatecodes) > 0:
        if namespace.updatecodes == "y":
            log_text = u"Обновление справочника кодов регионов..."
            print(log_text)
            write_log(log_text)
            saveCodeDictionary("regions.json")
            log_text = u"Обновление справочника завершено"
            print(log_text)
            write_log(log_text)
        elif namespace.updatecodes == "n":
            log_text = u"Обновление справочника отменено"
            print(log_text)
            write_log(log_text)
    if namespace.year is not None:
        year = namespace.year
    else:
        year = datetime.now().year
    if namespace.month is not None:
        months = [int(namespace.month)]
    else:
        if year == str(datetime.now().year):
            months = list(range(1, datetime.now().month, 1))
        else:
            months = list(range(1, 13, 1))
    filename = "regions.json"
    with codecs.open(filename, "r", "utf-8") as f:
        regions = json.load(f)
    getDTPInfo(data_root, year, months, regions, region_id=namespace.regcode)

if __name__ == "__main__":
    main()



