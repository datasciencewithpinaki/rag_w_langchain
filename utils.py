import pandas as pd
import numpy as np
from datetime import datetime
import re

from functools import wraps

# setting up logging
import logging
path_to_logfile = 'log_output/'
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(levelname)s: %(asctime)s: %(funcName)s: %(message)s')
file_handler = logging.FileHandler(path_to_logfile + 'log_utils.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# code starts here 

QUANTILES = [0.01, 0.03, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.97, 0.99]

def set_ipynb_config():
  from IPython.core.interactiveshell import InteractiveShell
  InteractiveShell.ast_node_interactivity = 'all'
    

def filter_data(df_raw:pd.DataFrame, dict_filter:dict)->pd.DataFrame:
    '''
    Filter the data wrt the dict {col: filter_criterion} that is passed.
    This also includes 'ALL' in the filter.
    Return back a filtered df.
    '''
    df = df_raw.copy()
    for k, v in dict_filter.items():
        if type(v)==list:
            if v==[]:
                continue
            filt = (df[k].isin(v))
        else:  # v is not a list
            if v=='ALL':
                continue
            filt = (df[k]==v)
        df = df.loc[filt,:]
    return df
  

def time_it(func, *args, **kwargs):
    '''
    Wrapper func that Returns time to execute a function in seconds
    '''
    @wraps(func)
    def call_func(*args, **kwargs):
        start_time = datetime.now()
        val = func(*args, **kwargs)
        end_time = datetime.now()
        exec_time = (end_time - start_time).total_seconds()
        logger.info(f'{str(func.__name__)} :: time taken to execute: {exec_time:.4f} seconds')
        return val
    return call_func
