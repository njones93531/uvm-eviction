import csv
import sys

max_time = 100000
default_threshold = 50
input_file_path = sys.argv[1]
output_file_path = sys.argv[1].split('.data')[0] + '.csv'
default_data = {'psize':'ERROR',
        'policy':'ERROR',
        'iter':'ERROR',
        'kernel_time':'0'}

#./matrixMul2 wA=31938 wB=31938 hA=31938 hB=31938 -p dmd -kernel_time 1 -r 0 -message thold 1 iter 0

#Problem Size,Policy,Iteration,Kernel Time
with open(input_file_path, 'r') as infile, open(output_file_path, 'w', newline='') as outfile:
    writer = csv.writer(outfile)
    writer.writerow(['Problem Size', 'Policy', 'Iteration', 'Kernel Time'])

    current_data = {}
    error_flag = False
    for line in infile:
        if line.startswith('./'):
            error_flag = False
            if current_data and current_data['psize'] != 'ERROR':
                writer.writerow([
                    current_data.get('psize', ''),
                    current_data.get('policy', ''),
                    current_data.get('iter', ''),
                    current_data.get('kernel_time', ''),
                ])
            current_data = default_data.copy()
            parts = line.strip().split(' ')
            current_data['psize'] = parts[1]
            for i, part in enumerate(parts):
                if(part) == "-p":
                    current_data['policy'] = parts[i+1]
                if(part) == 'iter' and len(parts) > i+1:
                    current_data['iter'] = parts[i+1]

        elif 'error' in line:
            parts = line.split()
            current_data['kernel_time'] = 0 
            error_flag = True

        elif 'GPU Runtime' in line:
            parts = line.split()
            current_data['kernel_time'] = parts[2].split('s')[0] if len(parts) > 2 else ''
            current_data['kernel_time'] = min(float(current_data['kernel_time']), max_time)
        elif 'timed' in line and ('h' in current_data['policy'] or 'm' in current_data['policy']) and (int(current_data['psize']) < 33 or 'd' not in current_data['policy']):
            if not error_flag: 
                parts = line.split()
                current_data['kernel_time'] = min(float(parts[4]), max_time)
            else:
                error_flag = False

    # Handle the last entry
    if current_data and current_data['psize'] != 'ERROR':
        writer.writerow([
                    current_data.get('psize', ''),
                    current_data.get('policy', ''),
                    current_data.get('iter', ''),
                    current_data.get('kernel_time', ''),
                ])

       
