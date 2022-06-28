# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 01:21:41 2022

@author: Jorge Castillo
"""

import pandas as pd
import re
from multiprocessing import cpu_count
from multiprocessing.dummy import Pool
from utils import is_decimal, extract_action, get_unique_alarms, get_time_statistics, \
                get_current_date, get_threshold_statistics
    
class AbstractFrame:
    def __init__(self, df):
        assert type(df) == pd.DataFrame
        self.df = df
        
    def __getitem__(self, idx):
        return self.df[idx]
        
    def __len__(self):
        return len(self.df)
    
    def __repr__(self):
        return self.df.to_string()

class ClinicFrame(AbstractFrame):
    def __init__(self, df):
        print('{} - {}'.format(get_current_date(), 'ClinicFrame'))
        super(ClinicFrame, self).__init__(df.dropna().reset_index(drop=True))
        
class EmptyBedFrame(AbstractFrame):
    def __init__(self, df):
        print('{} - {}'.format(get_current_date(), 'EmptyBedFrame'))
        super(EmptyBedFrame, self).__init__(df.iloc[:, :-1][df.iloc[:, :-1]['Bed Label'].isna()].reset_index(drop=True))
        
class EmptyDeviceFrame(AbstractFrame):
    def __init__(self, df):
        print('{} - {}'.format(get_current_date(), 'EmptyDeviceFrame'))
        super(EmptyDeviceFrame, self).__init__(df.iloc[:, :-1][df.iloc[:, :-1]['Device Name'].isna()].reset_index(drop=True))

class AlarmFrame(AbstractFrame):
    def __init__(self, df):
        super(AlarmFrame, self).__init__(df.iloc[:, :-1].dropna().reset_index(drop=True))
        
class GeneratedFrame(AlarmFrame):
    def __init__(self, df):
        super(GeneratedFrame, self).__init__(df)
        self.df = self.df.loc[self.df['Action'].str.contains("Generated")].reset_index(drop=True)
        
    def extract_dates(self, custom_indeces=None):
        if custom_indeces is None:
            custom_indeces = self.df.index
        dates = []
        for i, r in zip(self.df.index, self.df['Date']):
            original = r.split()
            original[1] = re.search(r'\d+:\d+:\d+', self.df['Action'][i]).group()
            dates.append(' '.join(original)) 
        return pd.DataFrame(dates, columns=['Date'], index=list(custom_indeces))
        
class EndedFrame(AlarmFrame):
    def __init__(self, df):
        super(EndedFrame, self).__init__(df)
        self.df = self.df.loc[self.df['Action'].str.contains("Ended.")].reset_index(drop=True)
        
class NurseFrame(AlarmFrame):
    def __init__(self, df):
        print('{} - {}'.format(get_current_date(), 'NurseFrame'))
        super(NurseFrame, self).__init__(df)
        ended = EndedFrame(df)
        generated = GeneratedFrame(df)
        self.df = self.df[~self.df['Action'].isin(ended['Action'])]
        self.df = self.df[~self.df['Action'].isin(generated['Action'])].reset_index(drop=True)
     
class ThresholdFrame(AbstractFrame):
    def __init__(self, df, target_action):
        assert type(df) == GeneratedFrame or type(df) == EndedFrame
        assert target_action is not None
        print('{} - {}: {} - Start'.format(get_current_date(), 'ThresholdFrame', target_action))
        if type(df) == GeneratedFrame:
            alarm_type = 'Generated'
        elif type(df) == EndedFrame:
            alarm_type = 'Ended'
        df = extract_action(df.df, target_action)
        super(ThresholdFrame, self).__init__(df)
        difference = []
        for a in df['Action']:
            action = a.split(alarm_type)[0].strip()#.replace('*', '\*')
            found = re.search(r'>|<', action)
            if found is not None:
                symbol = found.group()
                left = action.split(symbol)[0].split()[-1]
                right = action.split(symbol)[1]
                try:
                    difference.append(str(abs(float(left) - float(right))))
                except:
                    difference.append('-')
            else:
                difference.append('-')
        self.df = pd.concat([self.df.iloc[:, :-1], pd.DataFrame(difference, columns=['Difference']), self.df.iloc[:, -1]], axis=1) 
        print('{} - {}: {} - End'.format(get_current_date(), 'ThresholdFrame', target_action))
        
    def analyze_threshold(self):
        print(len(self.df))

class TimeFrame(AbstractFrame):
    def __init__(self, generated, ended, target_action=None):
        assert type(generated) == GeneratedFrame
        assert type(ended) == EndedFrame 
        assert target_action is not None
        print('{} - {}: {} - Start'.format(get_current_date(), 'TimeFrame', target_action))
        generated = extract_action(generated.df, target_action)
        ended = extract_action(ended.df, target_action)
        date = GeneratedFrame(generated).extract_dates(generated.index)
        gen_dates = pd.to_datetime(date['Date'])
        end_dates = pd.to_datetime(ended['Date'])
        
        used = []
        duration = []
        found_alarms = {}
        missing_alarms = {}
        for i, g in zip(ended.index, end_dates):
            action = ended['Action'][i].split('Ended')[0].strip()#.replace('*', '\*')
            found = re.search(r'>|<|:', action)
            dates = gen_dates.drop(labels=sorted(used))
            modified_gen = generated.drop(labels=sorted(used))
            self.df = modified_gen[dates < g]
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
            self.df = self.df[self.df['Action'].str.match(formatted_action)]
            self.df = self.df[self.df['Bed Label'].str.match(ended['Bed Label'][i])]
            self.df = self.df[self.df['Device Name'].str.match(ended['Device Name'][i])]
            exists = False
            if len(self.df) > 0:
                exists = True
                selected = self.df.index[0]
                time = int((end_dates[i] - pd.to_datetime(self.df['Date'][selected])).total_seconds())
                if time == 0:
                    try:
                        selected = self.df.index[1]
                        time = int((end_dates[i] - pd.to_datetime(self.df['Date'][selected])).total_seconds())
                    except:
                        exists = False
            if exists:
                used.append(selected)
                if formatted_action in found_alarms.keys():
                    found_alarms[formatted_action] += 1
                else:
                    found_alarms[formatted_action] = 1
                duration.append(str(time))
            else:
                if formatted_action in missing_alarms.keys():
                    missing_alarms[formatted_action] += 1
                else:
                    missing_alarms[formatted_action] = 1
                duration.append('-')
        
        self.found = found_alarms
        self.missing = missing_alarms
        self.df = pd.concat([ended.iloc[:, :-1], pd.DataFrame(duration, columns=['Duration'], index=list(ended.index)), ended.iloc[:, -1]], axis=1)
        print('{} - {}: {} - End'.format(get_current_date(), 'TimeFrame', target_action))

class ExperimentFrame(AbstractFrame):
    def __init__(self, df):
        assert type(df) == pd.DataFrame    
        # Original data
        self.data = df
        # Retrieve data with generated flag
        self.generated = self.get_generated()
        # Retrieve data with ended flag
        self.ended = self.get_ended()
        self.pool = Pool(cpu_count())
        self.df = { 'time': None, 'threshold': None, 'apnea': None, 'deep': None,
                    'clinic': None, 'bed': None, 'device': None, 'nurse': None }
        
    def get_clinic(self):
        self.df['clinic'] = ClinicFrame(self.data).df
        return self.df['clinic']
    
    def get_empty_bed(self):
        self.df['bed'] = EmptyBedFrame(self.data).df
        return self.df['bed']
    
    def get_empty_device(self):
        self.df['device'] = EmptyDeviceFrame(self.data).df
        return self.df['device']
    
    def get_nurse(self):
        self.df['nurse'] = NurseFrame(self.data).df
        return self.df['nurse']
    
    def get_generated(self):
        return GeneratedFrame(self.data)
    
    def get_ended(self):
        return EndedFrame(self.data)
    
    def get_time_difference(self, target_action):
        return TimeFrame(self.generated, self.ended, target_action)
    
    def get_threshold_difference(self, target_action):
        return ThresholdFrame(self.ended, target_action)
    
    def test(self, target_action):
        tf = self.get_threshold_difference(target_action)
        tf.analyze_threshold()
    
    def compute_deep_analysis(self, target_action):
        tf = self.get_threshold_difference(target_action)
        self.df['deep'] = { 'analysis': get_threshold_statistics(tf.df, deep=True), 
                             'data': tf.df }
        return self.df['deep']
    
    def compute_apnea_analysis(self, target_action='*** Apnea'):
        tf = self.get_time_difference(target_action)
        self.df['apnea'] = { 'analysis': get_time_statistics(tf.df, deep=True), 
                             'data': tf.df }
        return self.df['apnea']
        
    def compute_time_analysis(self, number_of_alarms=None):
        end_alarms = get_unique_alarms(self.ended.df)
        gen_alarms = get_unique_alarms(self.generated.df)
        
        if number_of_alarms is None:
            number_of_alarms = len(end_alarms)
        targets = []
        excluded = {}
        for i, e in enumerate(end_alarms.keys()):
            if e in gen_alarms.keys():
                targets.append(e)   
            else:
                excluded[e] = end_alarms[e]
            if i == number_of_alarms:
                break
        time_difference = self.pool.map(self.get_time_difference, targets)
        frames = [d.df for d in time_difference]
        self.df['time'] = { 'analysis': get_time_statistics(pd.concat(frames, axis=0).reset_index(drop=True), excluded),
                            'data': pd.concat(frames, axis=0).reset_index(drop=True) }
        return self.df['time']
        
    def compute_threshold_analysis(self):
        gen_alarms = get_unique_alarms(self.generated.df)
        targets = [e for e in gen_alarms.keys()]
        threshold_difference = self.pool.map(self.get_threshold_difference, targets)
        frames = [d.df for d in threshold_difference]
        analysis = {}
        for f in frames:
            unique = list(get_unique_alarms(f, False).keys())[0]
            stats = get_threshold_statistics(f, deep=True)
            if len(stats.keys()) != 0:
                analysis[unique] = stats
        self.df['threshold'] = { 'analysis': analysis,
                                 'data': frames }
        return self.df['threshold']
        
    def run_sample(self, number_of_alarms=None):
        # Retrieves data with non-empty clinical user field
        self.get_clinic()
        # Retrieve data with empty bed labels
        self.get_empty_bed()
        # Retrieve data with empty device labels
        self.get_empty_device()
        # Retrieve data with nurse actions
        self.get_nurse()
        # Basic analysis for thresholds for all alarms
        self.compute_threshold_analysis()
        # Basic analysis for timing for all alarms
        self.compute_time_analysis(number_of_alarms)
        # Changing the alarm will update the analysis
        self.compute_deep_analysis('**RR')
        # Similar to deep analysis but for Apnea alarms only
        # same as self.compute_deep_analysis('***Apnea')
        self.compute_apnea_analysis()

        


        
        