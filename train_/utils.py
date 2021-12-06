#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import argparse
 
def make_datetime_format(dt):
    dt = str(dt)
    if len(dt) == 8: return '-'.join([dt[:4], dt[4:6], dt[6:]])
 
 
def convert_data_split(data_split):
    try:
        data_split = float(data_split) if float(data_split) < 1 else data_split
    except:
        return data_split
     
    return data_split