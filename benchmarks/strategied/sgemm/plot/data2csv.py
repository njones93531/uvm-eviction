import csv
import sys

max_time = 100000

def parseElapsed(x):
    x = x.replace("elapsed","")
    x = x.split(':')
    y = float(x[0]) * 60 + float(x[1])
    return y


input_file_path = sys.argv[1]
output_file_path = "output.csv"
default_data = {'command':'ERROR',
        'policy':'ERROR',
        'iter':'ERROR',
        'aoi':'-1',
        'pressure':'0',
        'threshold':'50',
        'elapsed':'0',
        'timeout':'True'}

#./matrixMul2 wA=31938 wB=31938 hA=31938 hB=31938 -p dmd -aoi 1 -r 0 -message thold 1 iter 0
with open(input_file_path, 'r') as infile, open(output_file_path, 'w', newline='') as outfile:
    writer = csv.writer(outfile)
    writer.writerow(['Command', 'Policy', 'Iter', 'AOI', 'Pressure', 'Threshold', 'Elapsed', 'Timeout'])

    current_data = {}
    for line in infile:
        if line.startswith('./matrixMul2'):
            if current_data:
                writer.writerow([
                    current_data.get('command', ''),
                    current_data.get('policy', ''),
                    current_data.get('iter', ''),
                    current_data.get('aoi', ''),
                    current_data.get('pressure', ''),
                    current_data.get('threshold', ''),
                    current_data.get('elapsed', ''),
                    current_data.get('timeout', '')
                ])
            current_data = default_data
            parts = line.strip().split(' ')
            current_data['command'] = parts[0]
            for i, part in enumerate(parts):
                if(part) == "-p":
                    current_data['policy'] = parts[i+1]
                if(part) == '-aoi':
                    current_data['aoi'] = parts[i+1]
                if(part) == '-r':
                    current_data['pressure'] = parts[i+1]
                if(part) == 'thold':
                    current_data['threshold'] = parts[i+1]
                if(part) == 'iter':
                    current_data['iter'] = parts[i+1]
            if current_data['policy'] == 'hdd':
                current_data['aoi'] = 0
            if current_data['policy'] == 'dhd':
                current_data['aoi'] = 1
            if current_data['policy'] == 'ddh':
                current_data['aoi'] = 2
            if current_data['policy'] == 'mmm':
                current_data['aoi'] = -1
            if int(current_data['threshold']) < 10:
                current_data['threshold'] = '0' + current_data['threshold']

        elif 'user' in line:
            parts = line.split()
            current_data['elapsed'] = parseElapsed(parts[2]) if len(parts) > 2 else ''
            current_data['elapsed'] = min(float(current_data['elapsed']), max_time)
        elif 'inputs' in line:
            parts = line.split()
            current_data['timeout'] = 'Timeout' in line
        elif 'timed' in line:
            parts = line.split()
            current_data['elapsed'] = min(float(parts[4]), max_time)

    # Handle the last entry
    if current_data:
        writer.writerow([
                    current_data.get('command', ''),
                    current_data.get('policy', ''),
                    current_data.get('iter', ''),
                    current_data.get('aoi', ''),
                    current_data.get('pressure', ''),
                    current_data.get('threshold', ''),
                    current_data.get('elapsed', ''),
                    current_data.get('timeout', '')
                ])

       
