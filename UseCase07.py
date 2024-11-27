#
# Copyright (C) 2021 Transaction Processing Performance Council (TPC) and/or its contributors.
# This file is part of a software package distributed by the TPC
# The contents of this file have been developed by the TPC, and/or have been licensed to the TPC under one or more contributor
# license agreements.
# This file is subject to the terms and conditions outlined in the End-User
# License Agreement (EULA) which can be found in this distribution (EULA.txt) and is available at the following URL:
# http://www.tpc.org/TPC_Documents_Current_Versions/txt/EULA.txt
# Unless required by applicable law or agreed to in writing, this software is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied, and the user bears the entire risk as to quality
# and performance as well as the entire cost of service or repair in case of defect. See the EULA for more details.
#


#
# Copyright 2019 Intel Corporation.
# This software and the related documents are Intel copyrighted materials, and your use of them 
# is governed by the express license under which they were provided to you ("License"). Unless the 
# License provides otherwise, you may not use, modify, copy, publish, distribute, disclose or 
# transmit this software or the related documents without Intel's prior written permission.
# 
# This software and the related documents are provided as is, with no express or implied warranties, 
# other than those that are expressly stated in the License.
# 
#


#
import argparse
import os

# numerical computing
import timeit
from pathlib import Path

import numpy as np

# data frames
import pandas as pd

from surprise import SVD
from surprise import Dataset
from surprise.reader import Reader

import joblib


def load_data(path: str) -> pd.DataFrame:
    raw_data = pd.read_csv(path)
    # raw_data.columns = ['userID', 'productID', 'rating']
    return raw_data


def pre_process(data: pd.DataFrame) -> (np.array, np.array, Dataset):
    # data format appropriate for the surprise library
    if 'rating' in data.columns:
        data.drop_duplicates(['userID','productID'], keep='first', inplace=True)
    users = data.userID.unique()
    items = data.productID.unique()
    reader = Reader()
    if 'rating' in data.columns:
        data_transformed = Dataset.load_from_df(data[['userID', 'productID', 'rating']], reader)
    else:
        data_transformed = None
    return users, items, data_transformed


def train(data: Dataset) -> SVD:
    svd = SVD()
    trainset = data.build_full_trainset()
    model = svd.fit(trainset)
    return model


def serve(model, users, data,  n=None) -> pd.DataFrame:
    user_recommendations = []

    for u in users:
        ratings = []
        items = data[data['userID'] == u].productID
        for i in items:
            rating = model.predict(u, i).est
            ratings.append((u, i, rating))
        if n:
            ratings = sorted(ratings, key=lambda t: t[2], reverse=True)[:n]
        user_recommendations.extend(ratings)

    return pd.DataFrame(user_recommendations, columns=['userID', 'productID', 'rating'])


def main():
    model_file_name = "uc07.python.model"

    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true', required=False)
    parser.add_argument('--stage', choices=['training', 'serving', 'scoring'], metavar='stage', required=True)
    parser.add_argument('--workdir', metavar='workdir', required=True)
    parser.add_argument('--output', metavar='output', required=False)
    parser.add_argument("filename")

    args = parser.parse_args()
    path = args.filename
    stage = args.stage
    work_dir = Path(args.workdir)
    if args.output:
        output = Path(args.output)
    else:
        output = work_dir

    if not os.path.exists(work_dir):
        os.makedirs(work_dir)

    if not os.path.exists(output):
        os.makedirs(output)

    start = timeit.default_timer()
    raw_data = load_data(path)
    end = timeit.default_timer()
    load_time = end - start
    print('load time:\t', load_time)
    start = timeit.default_timer()
    (users, items, cleaned_data) = pre_process(raw_data)
    end = timeit.default_timer()
    pre_process_time = end - start
    print('pre-process time:\t', pre_process_time)

    if stage == 'training':
        start = timeit.default_timer()
        model = train(cleaned_data)
        end = timeit.default_timer()
        train_time = end - start
        print('train time:\t', train_time)
        joblib.dump(model, work_dir / model_file_name)

    if stage == 'serving':
        model = joblib.load(work_dir / model_file_name)
        start = timeit.default_timer()
        recommendations = serve(model, users, raw_data)
        end = timeit.default_timer()
        serve_time = end - start
        print('serve time:\t', serve_time)

        out_data = pd.DataFrame(recommendations)
        out_data.to_csv(output / 'predictions.csv', index=False)


if __name__ == '__main__':
    main()
