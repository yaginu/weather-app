import numpy as np
import pandas as pd
import tensorflow as tf

# Date列をtimesrtamp型に変換する
def convert_to_timestamp(data):

    date_time = pd.to_datetime(data)
    time_stamp = pd.Timestamp(date_time)

    return time_stamp.timestamp()

# wind_direction列を360°表記に変換する
def direction_to_degree(data):
    
    wind_direction = tf.strings.regex_replace(data, "[\s+)]", "")
    wind_direction = tf.strings.regex_replace(wind_direction, "[x]", "静穏")
    
    direction_list = [
        "北", "北北東", "北東", "東北東", "東", "東南東", "南東", "南南東", 
        "南", "南南西", "南西", "西南西", "西", "西北西", "北西", "北北西", "静穏"
    ]
    degree_list = [
        0.0, 22.5, 45.0, 67.5, 90.0, 112.5, 135.0, 157.5,
        180.0, 202.5, 225.0, 247.5, 270.0, 292.5, 315.0, 337.5, 0.0
    ]

    if data in direction_list:
        index = direction_list.index(data)
        return degree_list[index]
    else:
        return 0.0