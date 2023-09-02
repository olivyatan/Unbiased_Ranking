import datetime
import gzip
import subprocess
from functools import partial
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from pyspark.sql import Row

class HDFS:
    @staticmethod
    def copy_to_local(file_path, local_path):
        cmd = f'/apache/hadoop/bin/hadoop fs -copyToLocal {file_path} {local_path}'.split() # cmd must be an array of arguments
        return subprocess.call(cmd)

    @staticmethod
    def put(local_path, hdfs_path, force=True):
        if force:
            cmd = f'/apache/hadoop/bin/hadoop fs -put -f {local_path} {hdfs_path}'.split()
        else:
            cmd = f'/apache/hadoop/bin/hadoop fs -put {local_path} {hdfs_path}'.split()
        print(f'Upload command: {cmd}')
        return subprocess.call(cmd)

    @staticmethod
    def mkdir(hdfs_path, force=True):
        if force:
            cmd = f'/apache/hadoop/bin/hadoop fs -mkdir -p {hdfs_path}'.split()
        else:
            cmd = f'/apache/hadoop/bin/hadoop fs -mkdir {hdfs_path}'.split()

        return subprocess.call(cmd)


def unstack(arr, axis = 0):
    ar_lst = []
    for a in np.split(arr, arr.shape[0], axis=0):
        ar_lst.append(np.squeeze(a).tolist())
    return ar_lst


def process_path(idx, it, processor, base_path, partition_idx_col='fp_idx', with_partition_idx=False):
    def path_to_date(path):
        s = path.replace(f'{base_path}/', '').split('/')
        return f'{s[1]}-{s[2]}-{s[3]}'

    for path in it:
        data = load_npy_path(path)
        pdf = processor(data)
        if with_partition_idx:
            pdf[partition_idx_col] = idx
        
        pdf['dt'] = path_to_date(path)
                
        yield pdf


def load_npy_path(path):
    with TemporaryDirectory() as tmpDir:
        local_path = f'{tmpDir}/data.npy.gz'
        HDFS.copy_to_local(path, local_path)
        with open(local_path,'rb') as zf:
            with gzip.open(zf) as f:
                return np.load(f)


def npy_to_pdf(data, feature_cols=None, feature_col_prefix='f_'):
    if feature_cols is None:
        feature_cols = data['features'].dtype.names

    arr_feature_dict = {f'{feature_col_prefix}{tp[0]}': unstack(data['features'][tp[0]]) for tp in data['features'].dtype.descr if len(tp)==3  and tp[0] in feature_cols}
    features_dict = {f'{feature_col_prefix}{tp[0]}': data['features'][tp[0]] for tp in data['features'].dtype.descr if len(tp) <3 and tp[0] in feature_cols}
    labels_dict = {label: data['labels'][label] for label in data['labels'].dtype.names}
    cols_dict = {name: data[name] for name in data.dtype.names if name not in {'features', 'labels'}}
    data_dict = {**cols_dict, **labels_dict, **features_dict, **arr_feature_dict}
    return pd.DataFrame(data_dict)


def partition_to_rows(idx, it, feature_cols=None, feature_col_prefix='f_', with_partition_idx=True, partition_idx_col='fp_idx'):
    for path in it:
        data = load_npy_path(path)
        pdf = npy_to_pdf(data, feature_cols, feature_col_prefix)
        if with_partition_idx:
            pdf[partition_idx_col] = idx

        for row_idx, row in pdf.iterrows():
            yield Row(**row.to_dict())

            
class Fetcher:
    def __init__(self,   base_path, date_to_variant, start_date, end_date, fs, file_type='npy.gz', num_workers=64, partition_idx_col='fp_idx'):
        self.base_path = base_path
        self.date_to_variant = date_to_variant
        self.file_type = file_type
        self.num_workers = num_workers
        self.partition_idx_col = partition_idx_col
        self.partition_map = {}
        self.start_date, self.end_date = start_date, end_date
        self.fs = fs
        self.paths = self.calculate_paths(start_date, end_date)


    def extract_data_paths(self,  path_prefix, file_type):
        # fs = hdfs.HadoopFileSystem()
        paths = []
        walk = self.fs.walk(path_prefix)
        for f in walk:
            if len(f[2]) > 0:
                paths.extend([f'{f[0]}/{file}' for file in f[2] if file.endswith(f'.{file_type}')])
        return paths

    def fetch_pandas_df(self, spark, processor, with_partition_idx=False, to_spark=False):
        """
        Multiple executors apply processor to each data file (npy.gz).
        Outputs of all executors returned to driver and concatenated into a single pandas dataframe
        :param spark: spark session object
        :param processor: a method that receives numpy structured array object and returns pandas dataframe.
                        for example: lambda data: pd.DataFrame({'meid': data['meid'], 'labelClick': data['labels']['labelClick'], 'feature_name': data['features']['feature_name']})
        :return: pandas dataframe -- concatenation of processor results from all npy files.
        """
        rdd = spark.sparkContext.parallelize(self.paths, self.num_workers)
        results = rdd.mapPartitionsWithIndex(partial(process_path, processor=processor, partition_idx_col=self.partition_idx_col, with_partition_idx=with_partition_idx, base_path=self.base_path))
        
        return results if to_spark else pd.concat(results.collect())

            

    def fetch_spark_df(self, spark, feature_cols=None, feature_col_prefix='f_', with_partition_idx=False):
        rdd = spark.sparkContext.parallelize(self.paths, self.num_workers)
        mapped_rdd = rdd.mapPartitionsWithIndex(partial(partition_to_rows,
                                                        with_partition_idx=with_partition_idx,
                                                        partition_idx_col=self.partition_idx_col,
                                                        feature_cols=feature_cols,
                                                        feature_col_prefix=feature_col_prefix))
        return mapped_rdd.toDF()

    # def fetch_for_join(self, spark, start_date, end_date, cols=['meid', 'itemId']):
    #     processor = lambda data: pd.DataFrame({col: data[col] for col in cols}).drop_duplicates()
    #     pdf = self.fetch(spark, start_date, end_date, processor)
    #     if 'meid' in cols:
    #         pdf['meid'] = pdf.meid.map(lambda s: s.decode())
    #
    #     return spark.createDataFrame(pdf)
        
    def calculate_paths(self, start_date, end_date):
        dates = pd.date_range(start=start_date, end=end_date)
        # if pd.to_datetime('2023-03-31') in dates:
        #    dates = dates[dates != pd.to_datetime('2023-03-31')]
        data_paths = []
        for dt in dates:
            if dt in self.date_to_variant: 
                base_date_path = self.generate_path(dt)
                try:
                    data_paths.extend(self.extract_data_paths(base_date_path, file_type=self.file_type))
                except:
                    print('Error in path:', base_date_path)
        return data_paths 
    
    

    def generate_path(self, dt, suffix=''):
       
        algo_variant = self.date_to_variant[dt]
        return f'{self.base_path}/{algo_variant}/{dt.year}/{dt.month:02d}/{dt.day:02d}{suffix}'


def numpy_serializer(data, file):
    np.save(file, data)

    
class GzipHdfsUploader:
    @staticmethod
    def generate_timestamp_str():
        return datetime.datetime.now().strftime("%Y%m%d-%H%M")
    
    def __init__(self, base_out_path, base_src_path,  serializer=numpy_serializer, timestamp=None):
        self.base_out_path = base_out_path
        self.base_src_path = base_src_path
        self.timestamp = timestamp if timestamp is not None else self.generate_timestamp_str()
        self.serializer = serializer
        
    def __call__(self, data, path):
        suffix = path[len(self.base_src_path):]
        out_path = f'{self.base_out_path}/{self.timestamp}/{suffix}'
        out_dir_path = '/'.join(out_path.split('/')[:-1])
        file_name = path.split('/')[-1]
        
        HDFS.mkdir(out_dir_path, force=True)
        
        with TemporaryDirectory() as tmp_dir:
            local_path = f'{tmp_dir}/{file_name}'
            with gzip.open(local_path,'wb') as f:
                self.serializer(data, f)         
            
            return HDFS.put(local_path, f'{out_dir_path}/')


class UpdateProcessor:
    def __init__(self, join_columns):
        self.join_columns = join_columns
    
    def __call__(self, np_data, pdf):
        data_pdf = pd.DataFrame({col: np_data[col] for col in self.join_columns})

        if 'meid' in self.join_columns:
            data_pdf['meid'] = data_pdf.meid.map(lambda v: v.decode())

        filt_pdf = pd.merge(data_pdf, pdf, left_on=self.join_columns, right_on=self.join_columns, how='left')
        ext_feature_cols = [col for col in filt_pdf.columns if col not in self.join_columns]

        np_types = np_data.dtype.descr
        features_types = np_types[np_data.dtype.names.index('features')]

        # append new cols to schema
        for col in ext_feature_cols:
            features_types[1].append((col, '<f4'))
        # create empty array with the new size
        new_data = np.zeros(np_data.shape, dtype=np_types)

        # copy non-feature columns
        for col in np_data.dtype.names:
            if col != 'features':
                new_data[col] = np_data[col]
        # copy old feature columns
        for col in np_data.dtype[np_data.dtype.names.index('features')].names:
            new_data['features'][col]=np_data['features'][col]

        # copy new feature columns
        for col in ext_feature_cols:
            new_data['features'][col] = filt_pdf[col].astype('float32')
        

        del np_data
        del data_pdf
        
        return new_data
    

class Extender:
    def __init__(self, base_out_path, fetcher):
        self.base_out_path = base_out_path 
        self.partition_idx_col = fetcher.partition_idx_col
        self.partition_map = {idx: path for idx, path in enumerate(fetcher.paths)}
        self.src_base_path = fetcher.base_path

    def __call__(self, spark, df, join_columns=['meid', 'itemId'], num_workers=64):
        self.hdfs_uploader = GzipHdfsUploader(self.base_out_path, self.src_base_path)
        self.update_processor = UpdateProcessor(join_columns)
        
        rdd = df.rdd.map(lambda r: (r[self.partition_idx_col],r)).partitionBy(len(self.partition_map), lambda v: v)
        rdd.mapPartitionsWithIndex(self._run_update).collect()
        return f'{self.hdfs_uploader.base_out_path}/{self.hdfs_uploader.timestamp}'
            
    def _run_update(self, idx, it):
        rows = []
        for row in it:
            rows.append(row[1].asDict())
        
        if len(rows) == 0: 
            return None
        
        pdf = pd.DataFrame(rows)
        if self.partition_idx_col in pdf.columns:
            del pdf[self.partition_idx_col]
        
        print(f'External data columns: {pdf.dtypes}')
        src_file_path = self.partition_map[idx]
        
        data = load_npy_path(src_file_path)
        new_data = self.update_processor(data, pdf)
        
        yield self.hdfs_uploader(new_data, src_file_path)
        