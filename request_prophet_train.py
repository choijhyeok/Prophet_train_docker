import os
import sys
import argparse
import warnings
from configparser import ConfigParser 
from time import gmtime, strftime
import json
import requests
import random
import string
import boto3




def str2bool(v):
    if isinstance(v, bool): return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'): return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'): return False
    else: raise argparse.ArgumentTypeError('Boolean value expected.')


def send_request(args):
	# AWS key 
	boto_default_session = boto3.setup_default_session()
	boto_session = boto3.Session(botocore_session=boto_default_session, region_name="ap-northeast-2")
	credentials = boto_session.get_credentials()

	# RESTful 


	data = {}	  #dict 생성 해당하는 정보들을 담음
	data['aws_access_key_id'] = credentials.access_key 
	data['aws_secret_access_key'] = credentials.secret_key

	data['data_name'] = args.data_name
	data['project_name'] = args.project_name
	data['data_split'] = args.data_split
	data['bucket_name'] = args.bucket_name
	data['ds_col_name'] = args.ds_col_name
	data['y_col_name'] = args.y_col_name
	data['local_data_path'] = args.local_data_path
	data['save_model'] = args.save_model
	data['save_plot'] = args.save_plot
	data['data_encoding'] = args.data_encoding

	# print(data)

	# refer to this
	# https://curl.trillworks.com/#python
	headers = {'Content-Type': 'application/json',}
	url = 'http://' + args.host + ':' + args.port + '/receive'
	response = requests.post(url, headers=headers, data=json.dumps(data))


	if response.status_code == 200:
		print('prophet train has been successfully processed!')
	elif response.status_code == 404:
		print('Wrong Request Found.')

	# no longer store credentials
	data['aws_access_key_id'] =  ''.join(random.choice(string.digits+string.ascii_letters) for i in range(24))
	data['aws_secret_access_key'] =  ''.join(random.choice(string.digits+string.ascii_letters) for i in range(24))

	# dump this for other processes
	with open(args.json_prefix + '_prophet.json', 'w') as outfile:
		json.dump(data, outfile)

	# flushing
	data.clear()

if __name__ == '__main__':


	warnings.filterwarnings("ignore", category=FutureWarning)
	parser = argparse.ArgumentParser()
	
	parser.add_argument('--data-name', type=str, default='V3037_line_class_54-02.csv')
	parser.add_argument('--project_name', type=str, default= 'prophet-data-preprocessing-2021-11-30-06-28-52+0000')
	parser.add_argument('--data-split', type=str, default='0.8')
	parser.add_argument('--bucket_name', type=str,default='datascience-gsitm-cjh')
	parser.add_argument('--ds-col-name', type=str, default='datetime')
	parser.add_argument('--y-col-name', type=str, default='sales')
	parser.add_argument('--local-data-path', type=str, default='')
	parser.add_argument('--save-model', type=str2bool, default=True)
	parser.add_argument('--save-plot', type=str2bool, default=True)
	parser.add_argument('--data-encoding', type=str, default='')
	parser.add_argument('--json_prefix', type=str, default="this_path")
	parser.add_argument('--host', type=str, default='localhost')
	parser.add_argument('--port', type=str, default="50021")
	args = parser.parse_args()

    
	args, _ = parser.parse_known_args()
    
    
	send_request(args)
