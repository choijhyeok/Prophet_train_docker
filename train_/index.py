import os
import sys
import numpy as np
from flask import Flask, jsonify, request, Response, abort
import joblib
import pandas as pd
from io import StringIO
import json

import boto3
from boto3.session import Session
import botocore

from time import gmtime, strftime



from .train import *





app = Flask(__name__)


aws_credential_info = {"aws_access_key_id" : None, "aws_secret_access_key" : None}
repo_data_info  =  {"data_name" : None,
                    "project_name" : None,
                    "data_split" : None,
                    "bucket_name" : None,
                    "ds_col_name" : None,
                    "y_col_name" : None,
                    "local_data_path" : None,
                    "save_model" : None,
                    "save_plot" : None,
                    "data_encoding" : None}


def check_Nones(dict_file):
	value = 0
	for key in dict_file:
		if dict_file[key] == None:
			value = -1

	return value


@app.route('/receive', methods=['POST'])
def process_data():
	data = json.loads(request.data)

	aws_credential_info['aws_access_key_id'] = data.get("aws_access_key_id", None)
	aws_credential_info['aws_secret_access_key'] = data.get("aws_secret_access_key", None)

	repo_data_info['data_name'] = data.get("data_name", None)
	repo_data_info['project_name'] = data.get("project_name", None)
	repo_data_info['data_split'] = data.get("data_split", None)
	repo_data_info['bucket_name'] = data.get("bucket_name", None)
	repo_data_info['ds_col_name'] = data.get("ds_col_name", None)
	repo_data_info['y_col_name'] = data.get("y_col_name", None)
	repo_data_info['local_data_path'] = data.get("local_data_path", None)
	repo_data_info['save_model'] = data.get("save_model", None)
	repo_data_info['save_plot'] = data.get("save_plot", None)
	repo_data_info['data_encoding'] = data.get("data_encoding", None)
    


	check_stop_1 = check_Nones(aws_credential_info)
	check_stop_2 = check_Nones(repo_data_info)

	if check_stop_1 < 0 or check_stop_2 < 0:
		return abort(404)
	else:
		run_prophet(
	    aws_access_key = aws_credential_info['aws_access_key_id'],
		aws_secret_key = aws_credential_info['aws_secret_access_key'],
		data_name=repo_data_info['data_name'],
		data_split=repo_data_info['data_split'],
		project_name = repo_data_info['project_name'],
		bucket_name = repo_data_info['bucket_name'],
		ds_col_name=repo_data_info['ds_col_name'],
		y_col_name=repo_data_info['y_col_name'],
		local_data_path=repo_data_info['local_data_path'],
		save_model=repo_data_info['save_model'],
		save_plot=repo_data_info['save_plot'],
		data_encoding=repo_data_info['data_encoding'])

		#flushing
		aws_credential_info.clear()
		repo_data_info.clear()

		return jsonify("Process completed")

	

if __name__ == "__main__":
	app.run()
