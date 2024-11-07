from .base import AbstractDataset
from .utils import *

from datetime import date
from pathlib import Path
import pickle
import shutil
import tempfile
import os

import re
import numpy as np
import pandas as pd
from tqdm import tqdm
tqdm.pandas()


class MoocDataset(AbstractDataset):
    @classmethod
    def code(cls):
        return 'mooc_data'

    @classmethod
    def url(cls):  # as of Sep 2023
        return 'http://lfs.aminer.cn/misc/moocdata/data/course_recommendation.rar'

    @classmethod
    def zip_file_content_is_folder(cls):
        return True

    @classmethod
    def all_raw_file_names(cls):
        return ['Readme',
                'data.csv',
                'Data']

    def maybe_download_raw_dataset(self):
        folder_path = self._get_rawdata_folder_path()
        if folder_path.is_dir():
            return

        print("Raw file doesn't exist. Downloading...")
        tmproot = Path(tempfile.mkdtemp())
        tmpzip = tmproot.joinpath('file.rar')
        tmpfolder = tmproot.joinpath('folder')
        download(self.url(), tmpzip)
        print("download success")
        unrar(tmpzip, tmpfolder)
        if self.zip_file_content_is_folder():
            tmpfolder = tmpfolder.joinpath(os.listdir(tmpfolder)[0])
        shutil.move(tmpfolder, folder_path)
        shutil.rmtree(tmproot)
        print("unrar success")

    def preprocess(self):
        dataset_path = self._get_preprocessed_dataset_path()
        if dataset_path.is_file():
            print('Already preprocessed. Skip preprocessing')
            return
        if not dataset_path.parent.is_dir():
            dataset_path.parent.mkdir(parents=True)
        self.maybe_download_raw_dataset()
        df = self.load_ratings_df()
        meta_raw = self.load_meta_dict()
        df = df[df['sid'].isin(meta_raw)]  # filter items without meta info
        # df = self.filter_triplets(df)
        df, umap, smap = self.densify_index(df)
        train, val, test = self.split_df(df, len(umap))
        meta = {smap[k]: v for k, v in meta_raw.items() if k in smap}
        dataset = {'train': train,
                   'val': val,
                   'test': test,
                   'meta': meta,
                   'umap': umap,
                   'smap': smap}
        with dataset_path.open('wb') as f:
            pickle.dump(dataset, f)

    def load_ratings_df(self):
        folder_path = self._get_rawdata_folder_path()
        file_path = folder_path.joinpath('data.csv')
        df = pd.read_csv(file_path, encoding="GBK")
        df.columns = ['uid', 'timestamp', 'sid', 'course_name', "course_type", "type_id"]
        return df

    def load_meta_dict(self):
        folder_path = self._get_rawdata_folder_path()
        file_path = folder_path.joinpath('data.csv')
        df = pd.read_csv(file_path, encoding="GBK")
        meta_dict = {}
        for row in df.itertuples():
            course_index = row[3]
            name = row[4]
            meta_dict[course_index] = name
        return meta_dict
