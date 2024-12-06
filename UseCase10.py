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
# "INTEL CONFIDENTIAL" Copyright 2019 Intel Corporation All Rights
# Reserved.
#
# The source code contained or described herein and all documents related
# to the source code ("Material") are owned by Intel Corporation or its
# suppliers or licensors. Title to the Material remains with Intel
# Corporation or its suppliers and licensors. The Material contains trade
# secrets and proprietary and confidential information of Intel or its
# suppliers and licensors. The Material is protected by worldwide copyright
# and trade secret laws and treaty provisions. No part of the Material may
# be used, copied, reproduced, modified, published, uploaded, posted,
# transmitted, distributed, or disclosed in any way without Intel's prior
# express written permission.
#
# No license under any patent, copyright, trade secret or other
# intellectual property right is granted to or conferred upon you by
# disclosure or delivery of the Materials, either expressly, by
# implication, inducement, estoppel or otherwise. Any license under such
# intellectual property rights must be express and approved by Intel in
# writing.
#

import argparse
import os
import timeit

# data frames
from pathlib import Path
import pandas as pd

#logistic regression
from sklearn.linear_model import LogisticRegression

import joblib


def load_data(path_customers: str, path_transactions: str) -> pd.DataFrame:
    customer_data = pd.read_csv(path_customers)
    transaction_data = pd.read_csv(path_transactions)
    customer_data['senderID'] = customer_data['fa_customer_sk']
    data = pd.merge(transaction_data, customer_data, on="senderID")
    return (data)


def hour_func(ts):
    return ts.hour


def pre_process(data: pd.DataFrame) -> pd.DataFrame:
    data_pre = data
    data_pre['time'] = pd.to_datetime(data_pre['time'])
    data_pre['business_hour'] = data_pre['time'].apply(hour_func)
    data_pre['amount_norm'] = data_pre['amount'] / data_pre['transaction_limit']
    data_pre['business_hour_norm'] = data_pre['business_hour'] / 23
    if 'isFraud' in data_pre.columns:
        return data_pre[['transactionID', 'amount_norm', 'business_hour_norm', 'isFraud']]
    else:
        return data_pre[['transactionID', 'amount_norm', 'business_hour_norm']]


def train(data: pd.DataFrame) -> LogisticRegression:
    lrn = LogisticRegression(solver='lbfgs',C=1.0)
    X_train = data[["business_hour_norm", "amount_norm"]]
    y_train = data['isFraud']
    return lrn.fit(X_train, y_train)


def serve(model, data):
    data_serve = data[["business_hour_norm", "amount_norm"]]
    data['isFraud'] = model.predict(data_serve)
    return data[["transactionID", "isFraud"]]


def main():
    model_file_name = "uc10.python.model"

    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true', required=False)
    parser.add_argument('--stage', choices=['training', 'serving'], metavar='stage', required=True)
    parser.add_argument('--workdir', metavar='workdir', required=True)
    parser.add_argument('--output', metavar='output', required=False)
    parser.add_argument("customers")
    parser.add_argument("transactions")

    args = parser.parse_args()
    path_customers = args.customers
    path_transactions = args.transactions
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
    (raw_data) = load_data(path_customers, path_transactions)
    end = timeit.default_timer()
    load_time = end - start
    print('load time:\t', load_time)

    start = timeit.default_timer()
    preprocessed_data = pre_process(raw_data)
    end = timeit.default_timer()
    pre_process_time = end - start
    print('pre-process time:\t', pre_process_time)

    if stage == 'training':
        start = timeit.default_timer()
        model = train(preprocessed_data)
        end = timeit.default_timer()
        train_time = end - start
        print('train time:\t', train_time)

        joblib.dump(model, work_dir / model_file_name)

    if stage == 'serving':
        model = joblib.load(work_dir / model_file_name)
        start = timeit.default_timer()
        prediction = serve(model, preprocessed_data)
        end = timeit.default_timer()
        serve_time = end - start
        print('serve time:\t', serve_time)

        out_data = pd.DataFrame(prediction)
        out_data.to_csv(output / 'predictions.csv', index=False)


if __name__ == '__main__':
    main()
