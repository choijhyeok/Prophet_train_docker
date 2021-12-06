#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from numpy import not_equal
from logger import logger
from fbprophet import Prophet
from fbprophet.serialize import model_to_json, model_from_json
from fbprophet.plot import add_changepoints_to_plot
 
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
import os
 
matplotlib.use('agg')
sns.set_style('whitegrid')
 
IMAGE_TYPE = '.png'
IMAGE_DPI = 300
 

"""
plot_forecasting 에러 발생해서 수정했습니다.
model predict 부분에서 기존에는 test_data의 y값을 수정하지않고 넣었는데
그부분을 y 값을 제거후 넣는 방법으로 변경하였습니다.
"""
 
class AliceProphet(Prophet):
    def __init__(
        self,
        growth='linear',
        changepoints=None,
        n_changepoints=25,
        changepoint_range=0.8,
        yearly_seasonality='auto',
        weekly_seasonality='auto',
        daily_seasonality='auto',
        holidays=None,
        seasonality_mode='additive',
        seasonality_prior_scale=10,
        holidays_prior_scale=10,
        changepoint_prior_scale=0.05,
        mcmc_samples=0,
        interval_width=0.8,
        uncertainty_samples=1000,
        stan_backend=None,
        test_id=None,
        output_path=None,
        model_path=None):
         
        super().__init__(
            growth=growth,
            changepoints=changepoints,
            n_changepoints=n_changepoints,
            changepoint_range=changepoint_range,
            yearly_seasonality=yearly_seasonality,
            weekly_seasonality=weekly_seasonality,
            daily_seasonality=daily_seasonality,
            holidays=holidays,
            seasonality_mode=seasonality_mode,
            seasonality_prior_scale=seasonality_prior_scale,
            holidays_prior_scale=holidays_prior_scale,
            changepoint_prior_scale=changepoint_prior_scale,
            mcmc_samples=mcmc_samples,
            interval_width=interval_width,
            uncertainty_samples=uncertainty_samples,
            stan_backend=stan_backend)
         
        self.test_id = test_id
        self.output_path = output_path
        self.model_path = model_path
        self.metric_func = {}
     
 
    def __valid_path(self, _path: str) -> bool:
        if not _path.startswith('/'):
            current_path = os.getcwd()
            valid_path = os.path.join(current_path, _path)
        else:
            valid_path = _path
 
        return os.path.exists(valid_path)
 
     
    def save_model(self, model_path: str = None):
        model_path = model_path if model_path is not None else self.model_path
        if model_path is None: model_path = 'models'
        try:
            if not self.__valid_path(model_path): os.mkdir(model_path)
            with open(f'{model_path}/model.json', 'w') as f:
                json.dump(model_to_json(self), f)
 
        except Exception as e:
            print(f"Failed save model: {e}")
     
     
    def load_model(self, model_path=None):
        with open(f'{model_path}/model.json', 'r') as f:
            super = model_from_json(json.load(f))
 
         
    def save_metrics(self, output_path=None):
        output_path = output_path if output_path is not None else self.output_path
        if output_path is None: output_path = 'outputs'
        try:
            if not self.__valid_path(output_path): os.mkdir(output_path)
            with open(f'{output_path}/metrics.json', 'w') as f:
                json.dump(self.metrics, f)
 
        except Exception as e:
            print(f"Failed save metrics: {e}")
 
 
    def set_metrics(self, **kwargs) -> None:
        """
 
        """
        self.metric_func = kwargs
 
 
    def get_metrics(self) -> dict:
        return self.metrics
 
     
    def predict(self, test_data: pd.DataFrame, non_y_test_data : pd.DataFrame) -> pd.DataFrame:
        """
        """
        pred = super().predict(non_y_test_data)


        if len(self.metric_func):
            self.metrics = {metric: func(test_data['y'], pred['yhat']) for metric, func in self.metric_func.items()}
        return pred
 
 
    def plot_result(self,
        test_pred: pd.DataFrame,
        save=False,
        image_type=None,
        image_dpi=None,
        show=False):
        fix, ax = plt.subplots(figsize=(16, 5))
        super().plot(test_pred, ax=ax)
         
        if save:
            image_type = IMAGE_TYPE if image_type is None else image_type
            image_dpi = IMAGE_DPI if image_dpi is None else image_dpi
            plt.savefig(
                f'{self.output_path}/result{image_type}',
                dpi=image_dpi)
        if show: plt.show()
     
 
    def plot_components(self,
        fcst, uncertainty=True, plot_cap=True,
        weekly_start=0, yearly_start=0, figsize=None,
        save=False, image_type=None, image_dpi=None):
 
        super().plot_components(fcst, uncertainty=uncertainty, plot_cap=plot_cap, weekly_start=weekly_start, yearly_start=yearly_start, figsize=figsize)
        if save:
            image_type = IMAGE_TYPE if image_type is None else image_type
            image_dpi = IMAGE_DPI if image_dpi is None else image_dpi
            plt.savefig(
                f'{self.output_path}/components{image_type}',
                dpi=image_dpi)
 
 
    def plot_forecasting(self,
        df: pd.DataFrame,
        test_pred: pd.DataFrame,
        xlabel='date',
        ylabel=None,
        save=False,
        image_type=None,
        image_dpi=None,
        show=False):
        """
 
        """
        fig, ax = plt.subplots(figsize=(16,5))

        df = df.reset_index(drop=True)
        df['ds'] = pd.to_datetime(df['ds'])
 
        plt.plot(test_pred['ds'].dt.to_pydatetime(),test_pred['yhat'], label='forecast', color='blue')
        plt.plot(df['ds'].dt.to_pydatetime(), df['y'], label='observations ', color='black')
        plt.fill_between(test_pred['ds'].dt.to_pydatetime(), test_pred['yhat_upper'],test_pred['yhat_lower'],color='skyblue',label='80% confidence interval')
        plt.legend()
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
         
        if save:
            image_type = IMAGE_TYPE if image_type is None else image_type
            image_dpi = IMAGE_DPI if image_dpi is None else image_dpi
            plt.savefig(
                f'{self.output_path}/forecasting{image_type}',
                dpi=image_dpi)
        if show: plt.show()
 
 
    def plot_changepoint(self,
        test_pred,
        save=False,
        image_type=None,
        image_dpi=None):
        """
 
        """
        fig = self.plot(test_pred)
        a = add_changepoints_to_plot(fig.gca(), self, test_pred)
         
        if save:
            image_type = IMAGE_TYPE if image_type is None else image_type
            image_dpi = IMAGE_DPI if image_dpi is None else image_dpi
            plt.savefig(
                f'{self.output_path}/changepoints{image_type}',
                dpi=image_dpi)

