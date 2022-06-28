# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 02:22:26 2022

@author: Jorge Castillo
"""
import re
import pandas as pd
import numpy as np
from datetime import datetime

def get_current_date():
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    return dt_string

def is_decimal(value):
    if len(value.split('.')) > 1:
        return float(value)
    else:
        try:
            return int(value)
        except:
            return value
        
def evaluate_frame(df):
    assert type(df) == pd.DataFrame
    red_phys = df['Action'].str.contains('\*{3}\s*').sum()
    yellow_phys_double = df['Action'].str.contains('\*{2}').sum() - red_phys
    yellow_phys_single = df['Action'].str.contains('\*{1}').sum() - yellow_phys_double - red_phys
    
    red_inop = df['Action'].str.contains('!{3}').sum()
    yellow_inop_double = df['Action'].str.contains('!{2}').sum() - red_inop
    yellow_inop_single = df['Action'].str.contains('!{1}').sum() - red_inop - yellow_inop_double
    logged_inop = len(df) - red_phys - yellow_phys_double - yellow_phys_single - red_inop - yellow_inop_double - yellow_inop_single
    
    return {'Physiological': {'red': red_phys, 'yellow_double': yellow_phys_double, 'yellow_single': yellow_phys_single}, 
            'Inop': {'red': red_inop, 'yellow_double': yellow_inop_double, 'yellow_single': yellow_inop_single, 'logged': logged_inop}}

def extract_action(df, action):
    assert type(df) == pd.DataFrame
    found = re.search(r'>|<|:', action)
    if found is not None:
        symbol = found.group()
        value = is_decimal(action.split(symbol)[1].strip())
        alarm = ' '.join(action.split(symbol)[0].split()[:-1]).strip()
        if re.match('[*]+$', alarm) is not None:
            alarm = action.split(symbol)[0].strip()
        formatted_action = '{}.*{}.*{}'.format(alarm.replace('*', '\*'), symbol, value)
        if symbol == ':':
            value = is_decimal(action.split(symbol)[0].split()[-1].strip())
            formatted_action = '{}.*{}.*{}'.format(alarm.replace('*', '\*'), value, symbol)
    else:
        alarm = action
        formatted_action = alarm.replace('*', '\*')
    df = df[df['Action'].str.match(formatted_action)]  
    df = validate_frame(df, action)
    return df.reset_index(drop=True)

def validate_frame(df, target_action):
    assert type(df) == pd.DataFrame
    assert len(df) != 0
    # df = df.reset_index(drop=True)
    if len(df['Action'][df.index[0]].split('Generated')) > 1:
        alarm_type = 'Generated'
    else:
        alarm_type = 'Ended'
    valid = []
    for i, a in enumerate(df['Action']):
        action = a.split(alarm_type)[0].strip()
        found = re.search(r'>|<|:', action)
        if found is not None:
            symbol = found.group()
            alarm = ' '.join(action.split(symbol)[0].split()[:-1]).strip()
            if re.match('[*]+$', alarm) is not None:
                alarm = action.split(symbol)[0].strip()
        else:
            alarm = action
        if alarm == target_action:
            valid.append(i)
    return df.iloc[valid,:]

def get_unique_alarms(df, deep=False):
    assert type(df) == pd.DataFrame
    alarms = {}
    if len(df['Action'][df.index[0]].split('Generated')) > 1:
        alarm_type = 'Generated'
    else:
        alarm_type = 'Ended'
    for i, a in zip(df.index, df['Action']):
        action = a.split(alarm_type)[0].strip()#.replace('*', '\*')
        found = re.search(r'>|<|:', action)
        if found is not None and not deep:
            symbol = found.group()
            alarm = ' '.join(action.split(symbol)[0].split()[:-1]).strip()
            if re.match('[*]+$', alarm) is not None:
                alarm = action.split(symbol)[0].strip()
        else:
            alarm = action
        if alarm in alarms.keys():
            alarms[alarm]['count'] += 1
        else:
            alarms[alarm] = {}
            alarms[alarm]['count'] = 1
            alarms[alarm]['index'] = i
    return alarms

def get_threshold_statistics(df, deep=False):
    assert type(df) == pd.DataFrame
    alarms = {}
    unique = get_unique_alarms(df, deep).keys()
    for a in unique:
        found = re.search(r'<|>', a)
        if found is not None:
            symbol = found.group()
            threshold = a.split(symbol)[1].strip()
            filtered = df[df['Action'].str.match(a.replace('*', '\*'))]
            excluded = filtered[filtered['Difference'].str.contains('-')]
            difference = filtered[~filtered['Difference'].isin(excluded['Difference'])].reset_index(drop=True)
            if symbol == '>':
                value = float(threshold) + float(difference['Difference'][0])
            else:
                value = float(threshold) - float(difference['Difference'][0])
            if symbol+threshold in alarms.keys():
                alarms[symbol+threshold].append((value, len(filtered)))
            else:
                alarms[symbol+threshold] = [(value, len(filtered))]
            
            # values = {}
            # for i in range(len(difference)):
            #     if symbol == '>':
            #         value = float(threshold) + float(difference['Difference'][i])
            #     else:
            #         value = float(threshold) - float(difference['Difference'][i])
            #     if str(value) in values.keys():
            #         values[str(value)] += 1
            #     else:
            #         values[str(value)] = 1
            # alarms.append({'symbol': symbol, 'threshold': threshold, 
            #                'count': len(filtered), 'excluded': len(excluded),
            #                'values': values})
        # else:
        #     continue
    return alarms
    #     print('{} - {}: {}'.format(get_current_date(), 'GetThresholdStats', a))
    #     filtered = df[df['Action'].str.match(a.replace('*', '\*'))]
    #     excluded = filtered[filtered['Difference'].str.contains('-')]
    #     difference = filtered[~filtered['Difference'].isin(excluded['Difference'])]
    #     upper = difference[difference['Action'].str.contains('>')]
    #     lower = difference[difference['Action'].str.contains('<')]
    #     # invalid = abs(len(filtered) - len(difference))
    #     if len(upper) > 0 or len(lower) > 0:
    #         mean_upper = np.mean(upper['Difference'].astype(float))
    #         mean_upper = mean_upper if mean_upper is not np.nan else 0
    #         mean_lower = np.mean(lower['Difference'].astype(float))
    #         mean_lower = mean_lower if mean_lower is not np.nan else 0
            
    #         alarms[a] = {'occurrances <': len(lower), 'occurrances >': len(upper), 'excluded': len(excluded), 
    #                      'average <': mean_lower, 'average >': mean_upper}
    #     else:
    #         alarms[a] = {'occurrances <': 0, 'occurrances >': 0, 'excluded': len(excluded), 
    #                      'average <': 0, 'average >': 0}
    # return pd.DataFrame(list(alarms.values()), index=list(alarms.keys()))

def get_time_statistics(df, nonexistent={}, deep=False):
    assert type(df) == pd.DataFrame
    assert type(nonexistent) == dict
    alarms = {}
    unique = get_unique_alarms(df, deep).keys()
    excluded = df[df['Duration'].str.contains('-')]
    for a in unique:
        print('{} - {}: {}'.format(get_current_date(), 'GetTimeStats', a))
        filtered = df[df['Action'].str.match(a.replace('*', '\*'))]
        duration = filtered[~filtered['Duration'].isin(excluded['Duration'])]['Duration'].astype(int)
        invalid = abs(len(filtered) - len(duration))
        if len(duration) > 0:
            mean = np.mean(duration)
            std = np.std(duration)
            lower, upper = mean - 3*std, mean + 3*std
            filtered = list(filter(lambda s: s < upper, duration))
            filtered = list(filter(lambda s: s > lower, filtered))
            if len(filtered) == 0:
                filtered = duration
            # print(len(duration), len(filtered1), len(filtered))
            valid = len(filtered)
            invalid += abs(len(duration) - len(filtered))
            average_time = np.mean(filtered)
            max_time = max(filtered)
            min_time = min(filtered)
            p25, p50, p75 = np.percentile(duration, [25, 50, 75])
            alarms[a] = {'occurrances': valid, 'excluded': invalid, 'average': average_time, 'min': min_time, 'max': max_time, 
                          'P25': p25, 'P50': p50, 'P75': p75}
    for k in nonexistent.keys():
        alarms[k] = {'occurrances': 0, 'excluded': nonexistent[k]['count'], 'average': '-', 'min': '-', 'max': '-', 
                      'P25': '-', 'P50': '-', 'P75': '-'}
    return pd.DataFrame(list(alarms.values()), index=list(alarms.keys()))