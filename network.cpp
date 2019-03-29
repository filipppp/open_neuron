#include "network.h"
#include <array>
#include "layer.h"
#include "helpers/arrayhelper.h"
#include "matrix.h"
#include <iostream>

Network::Network(Layer** layers, unsigned long layerCount, double learningRate, unsigned short batchSize, float momentum,
	bool init) {
	this->layers = layers;
	this->learningRate = learningRate;
	this->batchSize = batchSize;
	this->momentum = momentum;
	this->layerCount = layerCount;

	for (unsigned long i = 1; i < layerCount; i++) {
		layers[i]->weights = (new Matrix(layers[i]->nodeCount, layers[i - 1]->nodeCount))->random();
		layers[i]->previousWeight = (new Matrix(layers[i]->nodeCount, layers[i - 1]->nodeCount, true));
		layers[i]->deltaWeightSum = Matrix::copy(layers[i]->previousWeight);
	}
}

double* Network::predict(double* inputs, unsigned long length) {
	if (length != layers[0]->nodeCount) { return nullptr; }

	// Set the input layer neurons to the passed inputs
	layers[0]->neurons = inputs;

	// Feed forward algorithm
	for (unsigned long i = 1; i < layerCount; i++) {
		layers[i]->neurons = Matrix::multiply(layers[i]->weights, layers[i - 1]->neurons, layers[i-1]->nodeCount)
			->add(layers[i]->biases, layers[i]->nodeCount)
			->apply(layers[i]->func);
	}
	printLastResult();
	// Return predicted output
	return layers[layerCount - 1]->neurons;
}

void Network::printLastResult() {
	for (unsigned long i = 0; i < layers[layerCount-1]->nodeCount; i++) {
		printf("reached");
		std::cout << layers[i]->neurons[i] << std::endl;
	}
}

double* Network::train(double* inputs, unsigned long inputLength, double* output, unsigned long outputLength) {
	return nullptr;
}
