import re 
import sys

toParse = open(sys.argv[1], "r")
relevantLines = []

patterns = ["ABC Migrate", "AB Migrate C Pin", "ABC Pin"]

benchmark = ""
for line in toParse:
    for pattern in patterns:
        r_pattern = re.compile("("+pattern+")")
        m = r_pattern.search(line)
        if m:
            print(line, end="")
