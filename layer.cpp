#include "layer.h"
#include "helpers/arrayhelper.h"

Layer::Layer(unsigned long nodeCount, ActivationFunction func) {
	neurons = ArrayHelper::getZero1D(nodeCount);
	biases = ArrayHelper::getZero1D(nodeCount);
	deltaBiasSum = ArrayHelper::getZero1D(nodeCount);
	this->nodeCount = nodeCount;
	this->func = func;
}

