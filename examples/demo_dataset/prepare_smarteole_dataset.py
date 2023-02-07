# Copyright 2021 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.


import os
from datetime import timedelta as td

import numpy as np
import pandas as pd
import requests

from zipfile import ZipFile

from floris.tools.floris_interface import FlorisInterface
from floris.utilities import wrap_360

from flasc.dataframe_operations import dataframe_manipulations as dfm
from flasc import floris_tools as ftls
from flasc import circular_statistics as circ

def download_smarteole_data():
    """Function that downloads 1-minute SCADA data from the SMARTEOLE wake
    steering experiment at the Sole du Moulin Vieux wind plant along with
    static wind plant and turbine data.
    """

    r = requests.get(r"https://zenodo.org/api/records/7342466#.Y-FG2S-B39A")

    r_json = r.json()

    filename = os.path.join("data",r_json["files"][0]["key"])

    result = requests.get(r_json["files"][0]["links"]["self"],stream=True)

    if not os.path.exists("data"):
        os.makedirs("data")

    if not os.path.exists(filename):
        print("SMARTEOLE data not found. Beginning file download from Zenodo...")

        chunk_number = 0

        with open(filename, "wb") as f:
            for chunk in result.iter_content(chunk_size=1024*1024):
        
                chunk_number = chunk_number + 1
        
                print(str(chunk_number) + " MB downloaded", end="\r")
        
                f.write(chunk)

    if not os.path.exists(filename[:-4]):
        print("Extracting SMARTEOLE zip file")
        with ZipFile(filename) as zipfile:
            zipfile.extractall("data")

def _compute_reference_variables(df):
    """Computes reference wind direction, wind speed, and power from four
    reference turbines and applies precomputed correction factors. 
    
    Args:
        df (pd.DataFrame): SMARTEOLE SCADA data frame with 1-minute data for
            all turbines.

    Returns:
        df [pd.DataFrame]: Dataframe with added wd, ws, and pow_ref.
    """

    # Load correction factors to apply to reference wind speed and power as a
    # function of wind speed and direction
    df_crct = pd.read_csv("data/SMARTEOLE-WFC-open-dataset/SMARTEOLE_WakeSteering_correction_factors_SMV1237_staticData.csv")

    # Load nacelle transfer function to correct reference wind speed to freestream
    df_ntf = pd.read_csv("data/SMARTEOLE-WFC-open-dataset/SMARTEOLE_WakeSteering_NTF_SMV6_staticData.csv")

    # Calculate reference wind direction, wind speed, and power as average of
    # turbines SMV 1, 2, 3, and 7
    df["wd"] = circ.calc_wd_mean_radial(df[[f"wind_direction_{ti}_avg" for ti in [1, 2, 3, 7]]],axis=1)
    df["ws"] = df[[f"wind_speed_{ti}_avg" for ti in [1, 2, 3, 7]]].mean(axis=1)
    df["pow_ref"] = df[[f"active_power_{ti}_avg" for ti in [1, 2, 3, 7]]].mean(axis=1)

    # Apply transfer functions to correct reference wind speed and power to 
    # match test turbine SMV6 in baseline operation. Note that corrections are
    # only provided for wind directions between 195 and 241 degrees, where wake
    # steering is analyzed.
    df["ws_round"] = df["ws"].round()
    df["wd_round"] = df["wd"].round()

    for i in range(len(df_crct)):
        wd = df_crct.iloc[0]["wind_direction_1237"]
        ws = df_crct.iloc[0]["wind_speed_1237"]
        df.loc[(df["wd_round"] == wd) & (df["ws_round"] == ws),"ws"] *= df_crct.iloc[0]["wind_speed_correction_factor_1237"]
        df.loc[(df["wd_round"] == wd) & (df["ws_round"] == ws),"pow_ref"] *= df_crct.iloc[0]["power_correction_factor_1237"]

    # Apply nacelle transfer function to correct reference wind speed to freestream
    df["ws"] = np.interp(df["ws"],df_ntf["wind_speed_6"],df_ntf["wind_speed_freestream"])

    # Drop temp columns
    df = df.drop(columns=["ws_round", "wd_round"])

    return df

if __name__ == "__main__":
    # Download SMARTEOLE SCADA data as well as wind plant and turbine data from Zenodo
    print("=================================================================")
    print("Downloading SMARTEOLE data")
    if not os.path.exists("data/SMARTEOLE-WFC-open-dataset"):
        df = download_smarteole_data()
    else:
        print("Data set already downloaded and extracted")

    # Combine 1-minute SCADA and control log files into single data frame
    df = pd.read_csv("data/SMARTEOLE-WFC-open-dataset/SMARTEOLE_WakeSteering_SCADA_1minData.csv")
    df_ctrl = pd.read_csv("data/SMARTEOLE-WFC-open-dataset/SMARTEOLE_WakeSteering_ControlLog_1minData.csv")
    df = df.merge(df_ctrl, how="inner", on="time")
    df["time"] = pd.to_datetime(df["time"])

    # Sort dataframe by time and fix duplicates
    df = dfm.df_sort_and_fix_duplicates(df)

    # Compute reference wind direction, wind speed, and power
    df = _compute_reference_variables(df)

    # Rename true 'wd' 'ws' channels
    df = df.rename(
        columns={"ws": "ws_truth", "wd": "wd_truth"}
    )

    root_path = os.path.dirname(os.path.abspath(__file__))
    fout = os.path.join(root_path, "smarteole_dataset_scada_60s.ftr")
    df.to_feather(fout)
    print("Saved SMARTEOLE SCADA dataset to: '" + fout + "'.")
