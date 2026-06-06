"""
Reads NRF log files and converts them into skate data (csv)

Defaults: 
Input:  ./Data/nrf_log.txt
Output: ./Data/skate_data.csv
"""

import re


def parse_nrf_log(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    readings = []
    current = {}
    
    # Only process lines starting with "A" (Application/received lines)
    for line in lines:
        if not line.strip().startswith('A'):
            continue
            
        # Extract "X:value" from lines like: A	13:40:45.940	"T:3308316" received
        match = re.search(r'"([TAG]):([^"]*)"', line)
        if not match:
            continue
            
        ptype, values = match.groups()
        
        if ptype == 'T':
            current = {'timestamp': values}
        elif ptype == 'A':
            p, y, r = values.split(',')
            current['pitch'] = p
            current['yaw'] = y
            current['roll'] = r
        elif ptype == 'G':
            ax, ay, az = values.split(',')
            current['accel_x'] = ax
            current['accel_y'] = ay
            current['accel_z'] = az
            
            # Complete set
            readings.append([
                current['timestamp'],
                current['pitch'], current['yaw'], current['roll'],
                current['accel_x'], current['accel_y'], current['accel_z']
            ])
    
    # Write CSV
    with open('./Data/skate_data.csv', 'w') as f:
        f.write('timestamp,pitch,yaw,roll,accel_x,accel_y,accel_z\n')
        for row in readings:
            f.write(','.join(row) + '\n')
    
    print(f"Parsed {len(readings)} readings")

parse_nrf_log('./Data/nrf_log.txt')