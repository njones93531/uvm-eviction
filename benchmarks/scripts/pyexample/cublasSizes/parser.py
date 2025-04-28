import re 
import sys

toParse = open(sys.argv[1], "r")
relevantLines = []

patterns = ["MatrixA.[0-9]*", "Performance= [0-9]*[.]*[0-9]*"]

benchmark = ""
for line in toParse:
    for pattern in patterns:
        r_pattern = re.compile("("+pattern+")")
        m = r_pattern.search(line)
        if m:
            match = m.group(1)
            if match[0:1] == "M":
                print(match[-5:].strip(),end=",")
                #benchmark = match[45:-6].strip()
                benchmark = "Voltron - 4GB Rel Mem"
            if match[0:1] == "P":
                print(match[12:].strip(),end=",")
                if benchmark == "AmovBmigCpin":
                    print("A Placed B Migrate C Pin")
                elif benchmark == "AmovBCmig":
                    print("A Placed BC Migrate")
                elif benchmark == "AmovCmigBpin":
                    print("A Placed C Migrate B Pin")
                elif benchmark == "AmovBCpin":
                    print("A Placed BC Pin")
                elif benchmark == "matmulABmigCpin":
                    print("AB Migrate C Pin")
                elif benchmark == "matmulACmigBpin":
                    print("AC Migrate B Pin")
                elif benchmark == "matmulDef":
                    print("ABC Migrate")
                elif benchmark == "matmulApinBCmig":
                    print("A Pin BC Migrate")
                elif benchmark == "matmulAmigBCpin":
                    print("A Migrate BC Pin")
                elif benchmark == "matmulABCpin":
                    print("ABC Pin")
                else: 
                    print(benchmark)
