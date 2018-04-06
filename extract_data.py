'''
Notes

Myoband data points unit = microseconds.

1 second is approximately ~200 data points
each data point has 8 values

EMG data in   ['emg']['data']   (list)
timestamps in ['emg']['timestamps']  (list)

index corresponds between two

'''


import json
from pprint import pprint

with open('first.json') as f:    
    all_data = json.load(f)


data = all_data['emg']['data']
timestamps = all_data['emg']['timestamps']

del all_data






