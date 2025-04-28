import sys

toParse = []
for i in range(1, len(sys.argv)):
    toParse.append(open(sys.argv[i], "r"))


csvs = []
for infile in toParse:
    lines = infile.readlines()
    for i, line in enumerate(lines):
        row = line.split(',')
        row[1] = str(float(row[1])/len(toParse))
        if len(csvs) < len(lines):
            csvs.append(row)
        else:
            csvs[i][1] = str(float(csvs[i][1]) + float(row[1]))


for row in csvs:
    for i, item in enumerate(row):
        if i < len(row) - 1:
            print(item, end=",")
        else: 
            print(item, end="")

for infile in toParse:
    infile.close()
