#include "layer.h"
#include "helpers/arrayhelper.h"
#include "activation/functions.h"

Layer::Layer(unsigned long nodeCount, Activation func) {
	neurons = ArrayHelper::getZero1D(nodeCount);
	biases = ArrayHelper::getZero1D(nodeCount);
	deltaBiasSum = ArrayHelper::getZero1D(nodeCount);
	this->nodeCount = nodeCount;
	this->func = func;
}

