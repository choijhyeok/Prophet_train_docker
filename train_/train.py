#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from .model import AliceProphet
from .datahandler import Repo, prepare_data, make_tarfile
from datetime import datetime
from .metrics import mean_absolute_percentage_error, root_mean_squared_error
from sklearn.metrics import mean_squared_error
from .utils import convert_data_split
 
import os
import shutil
import warnings
import argparse
import numpy as np
import multiprocessing
import pandas as pd


import sys
from flask import Flask, jsonify, request, Response, abort
import joblib
from io import StringIO
from time import gmtime, strftime



"""
기존의 코드에서 request 로 aws key를 받아서 repo에 전송시키는 역할과 
prepare_data의 코드 내용중에서 전처리 하는 과정을 데이터를 읽어온다음에 수행가능하도록 변경
run_train 부분에서는 큰 수정은 없고 경로들만 수정하였습니다.
"""



warnings.filterwarnings('ignore')
multiprocessing.set_start_method("fork")
 
 
METRIC_TABLE = {
    'mape': mean_absolute_percentage_error,
    'mse': mean_squared_error,
    'rmse': root_mean_squared_error}
 
PARAM_DICT = {
    'n_changepoints': int,
    'seasonality_mode': str,
    'seasonality_prior_scale': float,
    'holidays_prior_scale': float,
    'changepoint_prior_scale': float,
    'mcmc_samples': int,
    'interval_width': float,
    'uncertainty_samples': int}
 
 
def __get_params():
    return {p_name: p_type(os.getenv(p_name)) for p_name, p_type in PARAM_DICT.items() if os.getenv(p_name) is not None}
 
 
def run_train(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    project_name: str,
    save_model: bool = False,
    save_plot: bool = False,
    **kwargs):
    """
         
    """
    current_time = datetime.now().strftime("%Y-%m-%d:H%M%S")
    pro_name = project_name.split('/')[0]
    test_id = f'{pro_name}-{current_time}'
    model_path = os.path.join('models', test_id)
    output_path = os.path.join('outputs', test_id)
     
     
    if not os.path.exists('models'): os.mkdir('models')
    if not os.path.exists('outputs'): os.mkdir('outputs')
    os.mkdir(model_path)
    os.mkdir(output_path)
     
     
    # Prophet Model fitting.
    model = AliceProphet(
        output_path=output_path,
        model_path=model_path,
        test_id=test_id,
        **kwargs)
    

    model.add_country_holidays(country_name='KR')


    model.fit(train_data)
    model.set_metrics(**METRIC_TABLE)
    
    y_pred = test_data.drop('y',axis=1)

    test_pred = model.predict(test_data,y_pred)


    # return test_pred,model
     
    # Plotting & save test results.
    model.plot_result(test_pred, save=save_plot)
    model.plot_forecasting(pd.concat([train_data, test_data]), test_pred, save=save_plot)
    model.plot_components(test_pred, save=save_plot)
    model.plot_changepoint(test_pred, save=save_plot)
 
    # # Save model & metrics.
    if save_model: model.save_model()
    model.save_metrics()
     
    return {
        'id': test_id,
        'metrics': model.metrics,
        'result': 'complete',
        'outputs': output_path,
        'models' : model_path}
 
 
def run_prophet(
        aws_access_key : str,
        aws_secret_key : str,
        data_name: str,
        project_name : str,
        bucket_name : str,
        data_split: str or float,
        ds_col_name: str,
        y_col_name: str,
        local_data_path: str,
        save_model: bool,
        save_plot: bool,
        data_encoding: str = None):



   
   data_split = convert_data_split(data_split)
   upload_output = bool(int(os.getenv('ALICE_UPLOAD_OUTPUT', True)))
   remove_output = bool(int(os.getenv('ALICE_REMOVE_OUTPUT', True)))
   params = __get_params()
   
   print('')
   repo = Repo(
       project=project_name,
       bucket=bucket_name,
       aws_access_key = aws_access_key,
       aws_secret_key = aws_secret_key)
       
   train_data, test_data  = prepare_data(
        data_name=data_name,
        data_split=data_split,
        ds_col_name=ds_col_name,
        y_col_name=y_col_name,
        repo=repo,
        local_data_path=local_data_path,
        data_encoding=data_encoding)
    
   result = run_train(
        train_data=train_data,
        test_data=test_data,
        project_name=project_name,
        save_model=save_model,
        save_plot=save_plot,
        **params)
 
   test_id = result.get('id')
   output_path = result.get('outputs')
   model_path = result.get('models')

    
   if upload_output:
       f_name1 = f'{test_id}_plot_metrics.tar.gz'
       f_name2 = f'{test_id}_model.tar.gz'
       
       
       
       make_tarfile(output_filename = f_name1,source_dir=output_path,tar_name = 'plot_metrics')
       make_tarfile(output_filename = f_name2,source_dir=model_path,tar_name = 'model')

        
       repo.upload_outputs(output_path, f_name1,f'data/{repo.project}/outputs', other_path=False)
       repo.upload_outputs(model_path, f_name2,f'data/{repo.project}/models', other_path=False)



   if remove_output:
       shutil.rmtree('outputs')
       shutil.rmtree('models')
