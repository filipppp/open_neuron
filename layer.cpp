#include "layer.h"
#include "helpers/arrayhelper.h"
#include "activation/functions.h"
Layer::Layer(size_t nodeCount, Activation func) {
	neurons = ArrayHelper::getZero1D(nodeCount);

	biases = ArrayHelper::getZero1D(nodeCount);
	deltaBiasSum = ArrayHelper::getZero1D(nodeCount);
	previousDeltaBias = ArrayHelper::getZero1D(nodeCount);

	this->nodeCount = nodeCount;
	this->func = func;
}

Layer::~Layer() {
		delete[] biases;
		delete[] neurons;

		delete weights;
		delete deltaWeightSum;
		delete previousDeltaWeight;
		delete previousDeltaBias;

		delete[] deltaBiasSum;
}

void Layer::initWeights(Layer* previousLayer) {
	weights = (new Matrix(nodeCount, previousLayer->nodeCount))->random();
	previousDeltaWeight = (new Matrix(nodeCount, previousLayer->nodeCount, true));
	deltaWeightSum = Matrix::copy(previousDeltaWeight);
}

