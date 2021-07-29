import csv
import datetime
import calendar

from bs4 import BeautifulSoup
import requests

prec_no = [44]
block_no = [47662]
place_name = ["Tokyo"]

base_url = "http://www.data.jma.go.jp/obd/stats/etrn/view/hourly_s1.php?prec_no=%s&block_no=%s&year=%s&month=%s&day=%s&view=s1"

def str2float(str):
    try:
        return float(str)
    except:
        return 0.0
    
def fillna(str):
    if str is None:
        return "--"
    else:
        return str 

if __name__ == "__main__":

    for place in place_name:
        All_list = [['Date', 'air_pressure_ashore', 'air_pressure_afloat', 'precipitation',
                     'temperature', 'humidity', 'wind_velocity', 'wind_direction',
                     'hours_of_daylight', 'global_solar_radiation']]

        index = place_name.index(place)

        dt_now = datetime.datetime.now()
        this_year = dt_now.year
        this_month = dt_now.month
        yesterday = dt_now - datetime.timedelta(days=1)
        today = yesterday.day
        
        # year
        for year in range(2011, this_year+1):
            print(year)
            
            # month
            for month in range(1,13):
               
                # day
                # monthrangeで月の最終日を取得
                for day in range(1, calendar.monthrange(year, month)[1] + 1):
                    r = requests.get(base_url%(prec_no[index], block_no[index], year, month, day))
                    r.encoding = r.apparent_encoding

                    soup = BeautifulSoup(r.text)
                    rows = soup.findAll('tr', class_='mtx')
                    rows = rows[2:]

                    for row in rows:
                        data = row.findAll('td')

                        rowData = []

                        # Date
                        # datetime型は0~23時での表記なので、24時を翌日の0時に変換する
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

                        All_list.append(rowData)

                    if year == this_year and month == this_month and day == today:
                        break
                else:
                    continue
                break
            else:
                continue
            break

    with open(place + '.csv', 'w') as file:
                writer = csv.writer(file, lineterminator='\n')
                writer.writerows(All_list)