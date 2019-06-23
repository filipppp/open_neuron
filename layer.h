#ifndef layer_h
#define layer_h
#include "activation/functions.h"
#include "matrix.h"


class Layer {
public:

	Matrix* weights{};
	Matrix* deltaWeightSum{};
	Matrix* previousDeltaWeight{};

	double* biases;
	double* deltaBiasSum;
	
	double* neurons;

	Activation func;

	size_t nodeCount;

	Layer(size_t nodeCount,  Activation func);
	~Layer();

};

#endif
