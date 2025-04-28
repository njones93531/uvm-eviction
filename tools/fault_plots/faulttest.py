import sys
import numpy as np

batch_units = ['batches', 's', 'faults']
NS_PER_S = 1000000000

def get_range(addr, ranges):
    for i, (start, end) in enumerate(sorted(ranges, key=lambda tup: tup[0], reverse=True)):
        if start <= addr < end:
            return i

def batch_to_string(batch):
    string = ""
    for pair in zip(batch, batch_units):
        string += f'{pair[0]} {pair[1]}, '
    return string[:-1]

def get_avg_reoccurrence_batches(reoccurrences, total_reoccurrences, alloc='all'): 
    total_reoccurrence_batch = np.array([0.0,0.0,0.0])
    for fault in reoccurrences:
        if alloc == 'all' or reoccurrences[fault]['range'] == alloc:
            occurrences = [reoccurrences[fault]['first_occurred']] + reoccurrences[fault]['reoccurred']
            total_reoccurrence_batch += sum(occurrences[i+1] - occurrences[i] for i in range(0,len(occurrences)-1))
    return total_reoccurrence_batch / total_reoccurrences

def get_avg_offtime_batches(reoccurrences, total_reoccurrences, alloc='all'):
    total_offtime_batch = np.array([0.0,0.0,0.0])
    for fault in reoccurrences: 
        if alloc == 'all' or reoccurrences[fault]['range'] == alloc:
            for i in range(0, min(len(reoccurrences[fault]['reoccurred']), len(reoccurrences[fault]['evictions']))):
                total_offtime_batch += reoccurrences[fault]['reoccurred'][i] - reoccurrences[fault]['evictions'][i]
    return total_offtime_batch / total_reoccurrences

def print_avg_reoccurrence_vs_start(reoccurrences, total_reoccurrences, address_ranges):
    #avg num batches between first occurrence and reoccurance 
    if total_reoccurrences:
        avg_reoccurrence_batch = sum(sum(batch - reoccurrences[fault]['first_occurred'] for batch in reoccurrences[fault]['reoccurred']) for fault in reoccurrences) / total_reoccurrences
        print(f'Average interval between first fault and reoccurrence: {batch_to_string(avg_reoccurrence_batch)}')

        for i, (start, end) in enumerate(address_ranges):
            total_reoccurrences = sum(len(v['reoccurred']) if v['range'] == i else 0 for v in reoccurrences.values())
            if total_reoccurrences:
                avg_reoccurrence_batch = sum(sum(batch - reoccurrences[fault]['first_occurred'] for batch in reoccurrences[fault]['reoccurred']) for fault in reoccurrences if reoccurrences[fault]['range'] == i) / total_reoccurrences
                print(f'Average interval between first fault and reoccurrence in range {i}: {batch_to_string(avg_reoccurrence_batch)}')
            else:
                print(f"No fault reoccurrences in range {i}")
    else:
        print('Average interval between first fault and reoccurrence: N/A')


def print_avg_inter_reoccurrence_batches(reoccurrences, total_reoccurrences, address_ranges):
    #avg num batches between reoccurance and most recent occurance
    if total_reoccurrences:
        avg_reoccurrence_batch = get_avg_reoccurrence_batches(reoccurrences, total_reoccurrences)
        print(f'Average interval between fault reoccurrences: {batch_to_string(avg_reoccurrence_batch)}')

        for alloc, (start, end) in enumerate(address_ranges):
            total_reoccurrences = sum(len(v['reoccurred']) if v['range'] == alloc else 0 for v in reoccurrences.values())
            if total_reoccurrences:
                avg_reoccurrence_batch = get_avg_reoccurrence_batches(reoccurrences, total_reoccurrences, alloc)
                print(f'Average interval between fault reoccurrences in range {alloc}: {batch_to_string(avg_reoccurrence_batch)}')
            else:
                print(f"No fault reoccurrences in range {alloc}")
    else:
        print('Average interval between fault reoccurrences: N/A')


def print_avg_offtime_batches(reoccurrences, total_reoccurrences, address_ranges):
    #avg num batches between reoccurance and most recent eviction
    if total_reoccurrences:
        avg_offtime_batch = get_avg_offtime_batches(reoccurrences, total_reoccurrences)
        print(f'Average interval spent off device: {batch_to_string(avg_offtime_batch)}')

        for alloc, (start, end) in enumerate(address_ranges):
            total_reoccurrences = sum(len(v['reoccurred']) if v['range'] == alloc else 0 for v in reoccurrences.values())
            if total_reoccurrences:
                avg_offtime_batch = get_avg_offtime_batches(reoccurrences, total_reoccurrences, alloc)
                print(f'Average interval spent off device in range {alloc}: {batch_to_string(avg_offtime_batch)}')
            else:
                print(f"No fault reoccurrences in range {alloc}")

    else:
        print('Average interval spent off device: N/A')




def parse_file(filename):
    address_ranges = []
    batch = np.array([0.0,0.0,0.0])
    total_faults = 0
    total_evictions = 0
    total_faultless_evictions = 0
    first_fault_batch = {} # if it's in here it hasn't been evicted yet
    reoccurrences = {}  # Track reoccurrences of each fault, along with initial occurrence
    seen = {}
    evicted = {}


    with open(filename, 'r') as file:
        for line in file:
            parts = line.strip().split(',')

            if len(parts) == 2 and parts[0].startswith('0x'):
                start_address = int(parts[0], 16)
                length = int(parts[1])
                address_ranges.append((start_address, start_address + length))
            elif line.startswith('f,'):
                fault_address = int(parts[1], 16)
                total_faults += 1
                # should mean we've seen it before and it's been evicted
                if fault_address in seen and fault_address not in first_fault_batch:
                    # Record the batch difference since the first occurrence
                    if fault_address not in reoccurrences:
                        reoccurrences[fault_address] = {'first_occurred': seen[fault_address], 'reoccurred': [], 'range': get_range(fault_address, address_ranges), 'evictions': []}
                    if batch[0] != seen[fault_address][0]:
                        reoccurrences[fault_address]['reoccurred'].append(batch)
                    reoccurrences[fault_address]['evictions'].append(evicted[fault_address])

                seen[fault_address] = batch
                first_fault_batch[fault_address] = None

            elif line.startswith('b,'):
                batch = np.array([batch[0] + 1, (int(parts[1])/NS_PER_S) + batch[1], total_faults - batch[2]])
            elif line.startswith('e,'):
                total_evictions +=1
                evicted_address = int(parts[1], 16)
                aligned = (evicted_address // (2 * 1024 * 1024)) * (2 * 1024 * 1024)
                found = False
                for eva in range(evicted_address, evicted_address + 2 * 1024 * 1024, 4*1024):
                    if eva in first_fault_batch:
                        # Consider eviction as a reset for tracking
                        del first_fault_batch[eva]
                        evicted[eva] = batch
                        found = True
                    
                if not found:    
                    #print(f"Address {hex(evicted_address)} evicted without fault?")
                    total_faultless_evictions +=1

    print(f'# of batches: {batch[0]}')
    print(f'# of faults: {total_faults}')
    print(f'Average # of faults per batch: {total_faults / batch[0] if batch[0] else 0}')

    print(f'# of evictions: {total_evictions}')
    print(f'# of faultless evictions: {total_faultless_evictions}')
    print(f'% faultless evictions: {100.0 * float(total_faultless_evictions)/float(total_evictions)}')
    total_reoccurrences = sum(len(v['reoccurred']) for v in reoccurrences.values())
    print(f'# of faults that reoccur throughout the lifetime of the application: {total_reoccurrences}')

    
    print_avg_reoccurrence_vs_start(reoccurrences, total_reoccurrences, address_ranges)
    print_avg_inter_reoccurrence_batches(reoccurrences, total_reoccurrences, address_ranges)
    print_avg_offtime_batches(reoccurrences, total_reoccurrences, address_ranges)
    

    # Detailed Reoccurrence Summary
    #print("\nDetailed Reoccurrence Summary:")
    #for fault_address, details in reoccurrences.items():
        #print(f'Fault Address {hex(fault_address)} first occurred in batch {details["first_occurred"]} and reoccurred in batches: {details["reoccurred"]}')

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <file_path>")
    else:
        filename = sys.argv[1]
        parse_file(filename)

