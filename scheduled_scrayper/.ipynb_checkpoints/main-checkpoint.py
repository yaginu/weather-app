import csv
import datetime
import json

from bs4 import BeautifulSoup
import requests

from googleapiclient import discovery
from google.cloud import bigquery
from google.cloud import storage


# GCSへのファイルのアップロードを行う
def upload_blob(bucket_name, source_file_name, destination_blob_name, step):
    
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = "blob_{}".format(step)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    print(
        "File {} uploaded to {}.".format(
            source_file_name, destination_blob_name
        )
    )

# 数値特徴量の欠損値を0.0で埋める
def str2float(str):
    try:
        return float(str)
    except:
        return 0.0

# カテゴリー特徴量の欠損値を"--"で埋める
def fillna(str):
    if str is None:
        return "--"
    else:
        return str 

# datetime型はJSON形式で保存できないのでISO形式の文字列にに変換する
def json_serial(obj):
    if isinstance(obj, datetime.datetime):
        return obj.isoformat()
    raise TypeError ("Type {} not serializable".format(type(obj)))
    
# BigQueryへの書き込みをする
def table_insert_rows(dataset_id, table_id, rows_to_insert):
    
    client = bigquery.Client()
    table_ref = client.dataset(dataset_id).table(table_id)
    table = client.get_table(table_ref) 
    
    errors = client.insert_rows(table, rows_to_insert)
    if errors == []:
        print("New rows have been added.")
    else:
        print("Encounterd errors while inserting rows: {}".format(errors))

# BigQueryへのレコードの追加を行う
def insert_rows():    
    base_url = "http://www.data.jma.go.jp/obd/stats/etrn/view/hourly_s1.php?prec_no=%s&block_no=%s&year=%s&month=%s&day=%s&view=s1" 
    prec_no = [44]
    block_no = [47662]
    place_name = ["Tokyo"]
    
    dataset_id = "weather_data"
    table_id = "tokyo"
    
    for place in place_name:
        
        rows_to_insert = []
        prediction_data = []

        keys = ['Date', 'air_pressure_ashore', 'air_pressure_afloat', 'precipitation', 
                'temperature', 'humidity', 'wind_velocity', 'wind_direction',
                'hours_of_daylight', 'global_solar_radiation']
        
        index = place_name.index(place)
        
        dt_now = datetime.datetime.now()
        year = dt_now.year
        month = dt_now.month
        yesterday = dt_now - datetime.timedelta(days=1)
        day = yesterday.day
        print(year, month, day)
        
        r = requests.get(base_url%(prec_no[index], block_no[index], year, month, day))
        r.encoding = r.apparent_encoding
        
        soup = BeautifulSoup(r.text)
        rows = soup.findAll('tr', class_='mtx')
        rows = rows[2:]

        for row in rows:
            data = row.findAll('td')
            rowData = []

            if data[0].string == "24":
                rowData.append(datetime.datetime(year, month, day, 0) + datetime.timedelta(days=1))
            else:
                rowData.append(datetime.datetime(year, month, day, int(data[0].string)))

            rowData.append(str2float(data[1].string)) # air_pressure_ashore
            rowData.append(str2float(data[2].string)) # air_pressure_afloat
            rowData.append(str2float(data[3].string)) # precipitation
            rowData.append(str2float(data[4].string)) # temperature
            rowData.append(str2float(data[7].string)) # humidity
            rowData.append(str2float(data[8].string)) # wind_velocity
            rowData.append(fillna(data[9].string)) # wind_direction
            rowData.append(str2float(data[10].string)) # hours_of_daylight
            rowData.append(str2float(data[11].string)) # global_solar_radiation
            
            item = dict(zip(keys, rowData))            
            rows_to_insert.append(item)

            prediction_data.append(item)            

        table_insert_rows(dataset_id, table_id, rows_to_insert)

        with open('/tmp/daily_data.json', 'w') as f:
            json.dump(prediction_data, f, indent=4, default=json_serial)
            
        upload_blob("##########", # your backet
                    "/tmp/daily_data.json",
                    "staging/daily_predicion/{}/input".format(datetime.date.today().strftime("%Y-%m-%d")),
                    "json")
        
# カスタム予測ルーチンを使ったモデルは、バッチ予測ができないので、オンライン予測を使う
def predict_json(project, model, instances, version=None):

    service = discovery.build('ml', 'v1')
    name = 'projects/{}/models/{}'.format(project, model)

    if version is not None:
        name += '/versions/{}'.format(version)

    response = service.projects().predict(
        name=name,
        body={'instances': instances}
    ).execute()

    if 'error' in response:
        raise RuntimeError(response['error'])

    return response['predictions']

# 前日の気象データから当日の気温の予測を行い、結果をGCSに保存する
def saving_prediction():
    
    project_id = "######" # change your code
    model = "weather_forecast" 
    with open("/tmp/daily_data.json", mode="r") as f:
        instances = json.load(f)
        
    prediction = predict_json(project_id, model, instances)
    
    with open('/tmp/daily_prediction.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(prediction[0])
        
    upload_blob("##########", # your bucket
                "/tmp/daily_prediction.csv",
                "staging/daily_predicion/{}/prediction".format(datetime.date.today().strftime("%Y-%m-%d")),
                "csv")
    

def main(event, context):
    insert_rows()
    saving_prediction()