from typing import NamedTuple

def run_transformation_pipeline(
    source_table_name:str, job_name:str, gcs_root:str, project_id:str, region:str, dataset_location:str
) -> NamedTuple('Outputs', [('training_file_path', str), ('validation_file_path', str), ('testing_file_path', str), ('transform_artefacts_dir', str)]):
    
    import copy
    from datetime import datetime
    import math
    import os
    import tempfile

    from jinja2 import Template
    import apache_beam as beam

    import pandas as pd
    import tensorflow as tf    
    import tensorflow_transform as tft
    import tensorflow_transform.beam as tft_beam

        
    # Setting default value
    NUMERICAL_FEATURES = [
        'Date', 'air_pressure_ashore', 'air_pressure_afloat', 'precipitation', 'temperature',
        'humidity', 'wind_direction', 'wind_velocity', 'hours_of_daylight', 'global_solar_radiation'
    ]
    
    RAW_DATA_FEATURE_SPEC = dict(
        [(name, tf.io.FixedLenFeature([], tf.float32)) for name in NUMERICAL_FEATURES]
    )
    
    raw_metadata = tft.tf_metadata.dataset_metadata.DatasetMetadata(
        tft.tf_metadata.schema_utils.schema_from_feature_spec(RAW_DATA_FEATURE_SPEC)
    )    
    
    # Generating the query
    # train, valid, testセットのデータ量の比率が常に一定であるように、UNIX時間を使ってクエリを変化させる
    def generate_sampling_query(source_table_name, step):
        # Setting timestamp division
        start = datetime(2011, 1, 1, 1, 0, 0)
        end = datetime.now()
        diff = end.timestamp() - start.timestamp()

        train_start = start.timestamp()
        train_end = train_start + diff * 0.8
        valid_end = train_end + diff * 0.1
        test_end = valid_end + diff * 0.1

        train_start = datetime.fromtimestamp(train_start)
        train_end = datetime.fromtimestamp(train_end)
        valid_end = datetime.fromtimestamp(valid_end)
        test_end = datetime.fromtimestamp(test_end)

        valid_start = train_end
        test_start = valid_end

        # Template query
        sampling_query_template="""

    SELECT
        *
    FROM 
        `{{source_table}}`
    WHERE
        Date BETWEEN '{{start}}' AND '{{end}}'
    ORDER BY 
        Date

        """

        # Changing query dependging on steps
        if step == "Train":
            start, end = train_start, train_end
        elif step == "Valid":
            start, end = valid_start, valid_end
        else:
            start, end = test_start, test_end

        query = Template(sampling_query_template).render(
            source_table=source_table_name, start=start, end=end)

        return query
    
    def prep_bq_row(bq_row):
        
        result = {}
        
        for feature_name in bq_row.keys():
            result[feature_name] = bq_row[feature_name]
            
        date_time = pd.to_datetime(bq_row["Date"])
        time_stamp = pd.Timestamp(date_time)
        result["Date"] = time_stamp.timestamp()
        
        wind_direction = tf.strings.regex_replace(bq_row["wind_direction"], "[\s+)]", "")
        wind_direction = tf.strings.regex_replace(wind_direction, "[x]", u"静穏")  
        
        direction_list = [
            "北", "北北東", "北東", "東北東", "東", "東南東", "南東", "南南東", 
            "南", "南南西", "南西", "西南西", "西", "西北西", "北西", "北北西", "静穏"
        ]
        degree_list = [
            0.0, 22.5, 45.0, 67.5, 90.0, 112.5, 135.0, 157.5,
            180.0, 202.5, 225.0, 247.5, 270.0, 292.5, 315.0, 337.5, 0.0
        ]
        
        def direction_to_degree(direction):
            if direction in direction_list:
                index = direction_list.index(direction)
                return degree_list[index]
            else:
                return 0.0
            
        result["wind_direction"] = direction_to_degree(wind_direction)
        
        return result

    def read_from_bq(pipeline, source_table_name, step):
        
        query = generate_sampling_query(source_table_name, step)

        # Read data from Bigquery
        raw_data = (
            pipeline
            | 'Read{}DatafromBigQuery'.format(step) >> beam.io.Read(beam.io.ReadFromBigQuery(query=query, use_standard_sql=True))
            | 'Preproc{}Data'.format(step) >> beam.Map(prep_bq_row)
        )

        raw_dataset = (raw_data, raw_metadata)

        return raw_dataset

    def preprocess_fn(inputs):    
        outputs = {}
        
        # Date
        timestamp_s = inputs["Date"]
        
        day = 24 * 60 * 60
        year = 365.2425 * day
        
        outputs["day_sin"] = tf.sin(timestamp_s * 2 * math.pi / day)
        outputs["day_cos"] = tf.cos(timestamp_s * 2 * math.pi / day)
        
        outputs["year_sin"] = tf.sin(timestamp_s * 2 * math.pi / year)
        outputs["year_cos"] = tf.cos(timestamp_s * 2 * math.pi / year)
        
        # Air pressure
        STANDARDIZED_FEATURES_LIST = ["air_pressure_ashore", "air_pressure_afloat"]
        for feature in STANDARDIZED_FEATURES_LIST:
            outputs[feature] = tft.scale_to_0_1(tf.clip_by_value(inputs[feature], 860.0, 1100.0))

        outputs["diff_air_pressure"] = outputs["air_pressure_ashore"] - outputs["air_pressure_afloat"] 
        
        # Wind
        wind_direction_rad = inputs["wind_direction"] * math.pi / 180.0
        
        outputs["wind_vector_x"] = inputs["wind_velocity"] * tf.cos(wind_direction_rad)
        outputs["wind_vector_y"] = inputs["wind_velocity"] * tf.sin(wind_direction_rad)

        # Others
        # Normalizing numerical features
        NORMALIZED_FEATURES_LIST = ["precipitation", "temperature", "humidity", "hours_of_daylight", "global_solar_radiation"]
        for feature in NORMALIZED_FEATURES_LIST:
            outputs[feature] = tft.scale_to_z_score(inputs[feature])
        
        # Calcurating stats of Temperature and Converting to feature
        # preprocess_fn()で変換したデータに、trainセットのtemperature列の平均と分散が追加される
        def feature_from_scalar(value):
            batch_size = tf.shape(input=inputs["temperature"])[0]

            return tf.tile(tf.expand_dims(value, 0), multiples=[batch_size])
        
        outputs["temp_mean"] = feature_from_scalar(tft.mean(inputs['temperature']))
        outputs["temp_var"] = feature_from_scalar(tft.var(inputs['temperature']))
        
        return outputs

    def analyze_and_transform(raw_dataset, step):    

        transformed_dataset, transform_fn = (
            raw_dataset
            | tft_beam.AnalyzeAndTransformDataset(preprocess_fn)
        )

        return transformed_dataset, transform_fn

    def transform(raw_dataset, transform_fn, step):    

        transformed_dataset = (
            (raw_dataset, transform_fn)
            | '{}Transform'.format(step) >> tft_beam.TransformDataset()
        )

        return transformed_dataset
    
    def to_train_csv(rawdata):   
        
        TRAIN_CSV_COLUMNS = [
            'day_sin', 'day_cos', 'year_sin', 'year_cos', 'air_pressure_ashore', 'air_pressure_afloat', 'diff_air_pressure',
            'precipitation', 'temperature', 'humidity', 'wind_vector_x', 'wind_vector_y',
            'hours_of_daylight', 'global_solar_radiation', 'temp_mean', 'temp_var'
        ]

        data = ','.join([str(rawdata[k]) for k in TRAIN_CSV_COLUMNS])

        yield str(data)
    
    def to_test_csv(rawdata):

        TEST_CSV_COLUMNS = [
            'Date', 'air_pressure_ashore', 'air_pressure_afloat', 'precipitation', 'temperature',
            'humidity', 'wind_direction', 'wind_velocity', 'hours_of_daylight', 'global_solar_radiation'
        ]
        
        data = ','.join([str(rawdata[k]) for k in TEST_CSV_COLUMNS])

        yield str(data)

    def write_csv(transformed_dataset, location, step):    
        
        if step == "Train" or step == "Valid":
            transformed_data, _ = transformed_dataset
            (
                transformed_data
                | '{}Csv'.format(step) >> beam.FlatMap(to_train_csv)
                | '{}Out'.format(step) >> beam.io.Write(beam.io.WriteToText(location))
            )
        
        else:
            transformed_data, _ = transformed_dataset
            (
                transformed_data
                | '{}Csv'.format(step) >> beam.FlatMap(to_test_csv)
                | '{}Out'.format(step) >> beam.io.Write(beam.io.WriteToText(location))
            )

    def write_transform_artefacts(transform_fn, location):
        (
            transform_fn
            | 'WriteTransformArtefacts' >> tft_beam.WriteTransformFn(location)
            
        )
                       
    TRAINING_FILE_PATH = 'training/data.csv'
    VALIDATION_FILE_PATH = 'validation/data.csv'
    TESTING_FILE_PATH = 'testing/data.csv'
            
    options = {
        'staging_location': os.path.join(gcs_root, 'tmp', 'staging'),
        'temp_location': os.path.join(gcs_root, 'tmp'),
        'job_name': job_name,
        'project': project_id,
        'max_num_workers': 3,
        'save_main_session': True,
        'region': region,
        'setup_file': './setup.py', # pipeline内でtensorflow-transformを使えるようにsetup_fileの設定をする
    }
    
    opts = beam.pipeline.PipelineOptions(flags=[], **options)
    
    RUNNER = 'DataflowRunner'
    
    with beam.Pipeline(RUNNER, options=opts) as pipeline:
        with tft_beam.Context(temp_dir=tempfile.mkdtemp()):

            # Create training set
            step = "Train"            
            training_file_path = '{}/{}'.format(dataset_location, TRAINING_FILE_PATH)
            tf_record_file_path = dataset_location

            raw_train_dataset = read_from_bq(pipeline, source_table_name, step)
            transformed_train_dataset, transform_fn = analyze_and_transform(raw_train_dataset, step)            
            write_csv(transformed_train_dataset, training_file_path, step)
        
            # Create validation set
            step = "Valid"            
            validation_file_path = '{}/{}'.format(dataset_location, VALIDATION_FILE_PATH)

            raw_eval_dataset = read_from_bq(pipeline, source_table_name, step)
            transformed_eval_dataset = transform(raw_eval_dataset, transform_fn, step)
            write_csv(transformed_eval_dataset, validation_file_path, step)

            # Create testing set
            step = "Test"
            testing_file_path = '{}/{}'.format(dataset_location, TESTING_FILE_PATH)

            raw_test_dataset = read_from_bq(pipeline, source_table_name, step)
            write_csv(raw_test_dataset, testing_file_path, step)

            # Sarving artefacts
            transform_artefacts_dir = os.path.join(gcs_root,'transform') 
            write_transform_artefacts(transform_fn, transform_artefacts_dir)
                        
    return (training_file_path, validation_file_path, testing_file_path, transform_artefacts_dir)