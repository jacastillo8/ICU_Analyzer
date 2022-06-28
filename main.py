# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 13:18:53 2022

@author: Jorge Castillo
"""

import pandas as pd
import sys
sys.path.append('./lib')
from lib.medframes import ExperimentFrame

if __name__ == '__main__':  
    original = pd.read_csv('./data/data.csv')
    experiment = ExperimentFrame(original)
    # Retrieves data with non-empty clinical user field
    clinic_frame = experiment.get_clinic()
    # Retrieve data with empty bed labels
    empty_bed_frame = experiment.get_empty_bed()
    # Retrieve data with empty device labels
    empty_device_frame = experiment.get_empty_device()
    # Retrieve data with nurse actions
    nurse_action_frame = experiment.get_nurse()
    # Basic analysis for thresholds for all alarms
    threshold_analysis_frame = experiment.compute_threshold_analysis()
    # Basic analysis for timing for all alarms
    time_analysis_frame = experiment.compute_time_analysis(number_of_alarms=None)
    # Changing the alarm will update the analysis
    single_alarm_analysis_frame = experiment.compute_deep_analysis('**RR')
    # Similar to deep analysis but for Apnea alarms only
    # same as self.compute_deep_analysis('***Apnea')
    apnea_analysis_frame = experiment.compute_apnea_analysis()

    