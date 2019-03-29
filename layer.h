#ifndef layer_h
#define layer_h

class Matrix;

enum ActivationFunction {
	sigmoid,
	relu,
	identity
};

class Layer {
public:

	Matrix* weights;
	Matrix* deltaWeightSum;
	Matrix* previousWeight;

	double* biases;
	double* deltaBiasSum;
	
	double* neurons;

	ActivationFunction func;

	unsigned long nodeCount;
	 
	Layer(unsigned long nodeCount,  ActivationFunction func = sigmoid);

};

#endif
