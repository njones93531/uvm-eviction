#ifndef PROB_H
#define PROB_H

#include <stdio.h>
#include <stdlib.h>

struct prob {
	int* weights;
	int* values;
	
	int num_items;

	int weight_cap;
};

struct prob read_prob(char *filename, int num_items);

#endif 
