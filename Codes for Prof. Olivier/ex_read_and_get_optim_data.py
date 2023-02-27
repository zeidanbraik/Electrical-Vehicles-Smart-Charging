# -*- coding: utf-8 -*-
"""
Created on Thursday Jan 12 18:33:53 2023

@author: Olivier BEAUDE
"""
import datetime
import os

import numpy as np
import pandas as pd


def get_data_from_eco2mix_file(data_directory: str, year: int = 2019, season: str = "summer") -> pd.DataFrame:
    """
    Get useful data from a given Eco2mix file

    :param data_directory: directory in which data is located
    :param year: year to be considered, to choose the proper Eco2mix file
    :param season: idem, for the season
    """
    # define the name and sheet where Eco2Mix data have to be read
    eco2mix_file = f"eCO2mix_RTE_Annuel-Definitif_{year}_{season}.csv"

    # read data in Eco2mix .csv file, and obtain it as pandas dataframe
    df_eco2mix = pd.read_csv(os.path.join(data_directory, eco2mix_file), sep=";", decimal=".", encoding="ANSI")

    # convert data column to datetime objects
    df_eco2mix["date"] = df_eco2mix["date"].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))

    return df_eco2mix


def get_nonflex_load(df_eco2mix: pd.DataFrame, period_start: datetime.datetime,
                     period_end: datetime.datetime) -> np.ndarray:
    """
    Get non-flexible load profile for a given time period
    :param df_eco2mix: pandas dataframe containing Eco2mix data
    :param period_start: start of the period to be considered
    :param period_end: idem, end
    :return: numpy array containing non-flexible load data for considered period
    """

    df_eco2mix = df_eco2mix.loc[(df_eco2mix["date"] >= period_start) & (df_eco2mix["date"] <= period_end)]

    return np.array(df_eco2mix["forecast_day-1"])


if __name__ == "__main__":
    # get current dir. full path
    current_dir = os.getcwd()
    current_year = 2019
    current_season = "summer"
    delta_t_h = 0.5  # time-slot duration, in hour

    # set directory in which data is located
    data_dir = os.path.join(current_dir, "data")
    df_eco2mix = get_data_from_eco2mix_file(data_directory=data_dir, year=current_year, season=current_season)

    # get non-flexible load from Eco2mix dataframe
    ts_arr = 35
    ts_dep = 14
    current_day = datetime.datetime(2019, 6, 1)
    date_arr = current_day + datetime.timedelta(hours=ts_arr * delta_t_h)
    date_dep = current_day + datetime.timedelta(days=1) + datetime.timedelta(hours=ts_dep * delta_t_h)
    np_nonflex_load = get_nonflex_load(df_eco2mix=df_eco2mix, period_start=date_arr, period_end=date_dep)