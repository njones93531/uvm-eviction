import sys
import random

def write(arr):
	g = open("prob.csv", "w")

	for item in arr:
		g.write(str(item[0]) + "," + str(item[1]) + "\n")

	g.close()

if __name__ == "__main__":
	if len(sys.argv) == 1:
		num_items = 10
	else:
		num_items = int(sys.argv[1])

	weight_val = []

	for i in range(num_items):
		weight_val.append([random.randint(100, 500),
				random.randint(1, 10)])

	print(weight_val)
	
	write(weight_val)
