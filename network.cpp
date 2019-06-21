#include "network.h"
#include <array>
#include "layer.h"
#include "helpers/arrayhelper.h"
#include "matrix.h"
#include <iostream>
#include "../../../../../../Program Files (x86)/Microsoft Visual Studio/2019/Enterprise/VC/Tools/MSVC/14.21.27702/include/ctime"

Network::Network(Layer** layers, unsigned long layerCount, double learningRate, unsigned short batchSize, float momentum,
	bool init) {
	srand(time(NULL));

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

Network::~Network()
{
	for (int i = 0; i < layerCount; ++i)
	{
		delete layers[i];
	}
	delete[] layers	;
}

double* Network::predict(double* inputs, unsigned long length)
{
	if (length != layers[0]->nodeCount) { return nullptr; }

	/* Set the input layer neurons to the passed inputs */
	layers[0]->neurons = inputs;

	/* Feed forward algorithm */
	for (unsigned long i = 1; i < layerCount; i++) {
		/*
		 * 1. Multiply previous Neurons with current index I weights to get new Neuron Matrix for index I
		 * 2. add() bias to calclulated neurons
		 * 3. apply() Activation Function to squeeze numbers to specific regions
		 */
		layers[i]->neurons = Matrix::multiply(layers[i]->weights, layers[i - 1]->neurons, layers[i-1]->nodeCount)
			->add(layers[i]->biases, layers[i]->nodeCount)
			->apply(layers[i]->func);
	}


	/* Return predicted output which are just the last neurons */
	return layers[layerCount - 1]->neurons;
}

void Network::printLastResult() {
	for (unsigned long i = 0; i < layers[layerCount-1]->nodeCount; i++) {
		std::cout << layers[layerCount - 1]->neurons[i] << std::endl;
	}
}

void Network::train(double* input, unsigned long inputLength, double* targetOutput, unsigned long targetOutputLength) {
	if (targetOutputLength != layers[layerCount - 1]->nodeCount) { 
		std::cout << "wrong training data";
		return; 
	}
	/* Calculate Error which Network gives for specific input */
	double* predictedOutput = predict(input, inputLength);
	double* outputError = ArrayHelper::subtractArrays(targetOutput, predictedOutput, targetOutputLength);

	double** errorMatrices = new double*[layerCount];
	errorMatrices[layerCount - 1] = outputError;

	/* Iterating through layers from behind */
	for (unsigned long i = layerCount - 1; i > 0; i--)
	{
		double* neuronsDerivative = ArrayHelper::mapTo(layers[i]->neurons, layers[i]->nodeCount, layers[i]->func, true);
		double* gradient = ArrayHelper::multiply(ArrayHelper::hadamardArray(neuronsDerivative, errorMatrices[i], layers[i]->nodeCount), learningRate, layers[i]->nodeCount);
		
		Matrix* deltaWeights = Matrix::multiply(gradient, layers[i]->nodeCount, layers[i - 1]->neurons, layers[i - 1]->nodeCount);

		layers[i]->weights->add(deltaWeights);
		ArrayHelper::add(layers[i]->biases, gradient, layers[i]->nodeCount);

		errorMatrices[i - 1] = Matrix::multiply(Matrix::transpose(layers[i]->weights), errorMatrices[i], layers[i]->nodeCount)->to1d();
	}
}
