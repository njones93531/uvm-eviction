import re 
import sys

toParse = open(sys.argv[1], "r")
relevantLines = []

patterns = ["Running.*\n", "Time total................................[0-9]*.[0-9]*"]

benchmark = ""
for line in toParse:
    for pattern in patterns:
        r_pattern = re.compile("("+pattern+")")
        m = r_pattern.search(line)
        if m:
            match = m.group(1)
            if match[0:1] == "R":
                print(match[-6:].strip(),end=",")
                benchmark = match[-10:-7].strip()
            if match[0:1] == "T":
                print(match[41:].strip(),end=",")
                print(benchmark)
