import re 
import sys

toParse = open(sys.argv[1], "r")
relevantLines = []

patterns = ["Running.*\n", "Runtime: [0-9]*.[0-9]*"]

benchmark = ""
for line in toParse:
    for pattern in patterns:
        r_pattern = re.compile("("+pattern+")")
        m = r_pattern.search(line)
        if m:
            match = m.group(1)
            if match[3:4] == "n":
                print(match[-6:].strip(),end=",")
                benchmark = match[match.find("_vanilla_")+9:-7].strip()
            if match[3:4] == "t":
                print(match[9:].strip(),end=",")
                print(benchmark)
