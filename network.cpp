#include "network.h"
#include <array>
#include "layer.h"
#include "helpers/arrayhelper.h"
#include "matrix.h"
#include <iostream>
#include "../../../../../../Program Files (x86)/Microsoft Visual Studio/2019/Enterprise/VC/Tools/MSVC/14.21.27702/include/ctime"

Network::Network(Layer** layers, size_t layerCount, double learningRate, unsigned short batchSize, float momentum,
	bool init) {
	srand(time(NULL));

	this->layers = layers;
	this->learningRate = learningRate;
	this->batchSize = batchSize;
	this->momentum = momentum;
	this->layerCount = layerCount;

	for (size_t i = 1; i < layerCount; i++) {
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
	// delete[] layers;
}

double* Network::predict(double* inputs, size_t length)
{
	if (length != layers[0]->nodeCount) {
		return nullptr;
	}

	/* Set the input layer neurons to the passed inputs */
	for (size_t i = 0; i < length; ++i) {
		layers[0]->neurons[i] = inputs[i];
	}

	/* Feed forward algorithm */
	for (size_t i = 1; i < layerCount; i++) {
		/*
		 * 1. Multiply previous Neurons with current index I weights to get new Neuron Matrix for index I
		 * 2. add() bias to calclulated neurons
		 * 3. apply() Activation Function to squeeze numbers to specific regions
		 */
		delete[] layers[i]->neurons;
		Matrix* unmappedMatrix = Matrix::multiply(layers[i]->weights, layers[i - 1]->neurons, layers[i - 1]->nodeCount)
			->add(layers[i]->biases, layers[i]->nodeCount);
		layers[i]->neurons = unmappedMatrix->apply(layers[i]->func);

		delete unmappedMatrix;
	}

	/* Return predicted output which are just the last neurons */
	return layers[layerCount - 1]->neurons;
}

void Network::printLastResult() {
	for (size_t i = 0; i < layers[layerCount-1]->nodeCount; i++) {
		std::cout << layers[layerCount - 1]->neurons[i] << std::endl;
	}
}

void Network::train(double* input, size_t inputLength, double* targetOutput,size_t targetOutputLength) {
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
	for (size_t i = layerCount - 1; i > 0; i--) {
		double* neuronsDerivative = ArrayHelper::mapTo(layers[i]->neurons, layers[i]->nodeCount, layers[i]->func, true);

		double* errorMultNeurons = ArrayHelper::hadamardArray(neuronsDerivative, errorMatrices[i], layers[i]->nodeCount);
		double* gradient = ArrayHelper::multiply(errorMultNeurons, learningRate, layers[i]->nodeCount);

		delete[] neuronsDerivative;
		delete[] errorMultNeurons;

		Matrix* deltaWeights = Matrix::multiply(gradient, layers[i]->nodeCount, layers[i - 1]->neurons, layers[i - 1]->nodeCount);
		layers[i]->weights->add(deltaWeights);
		delete deltaWeights;

		ArrayHelper::add(layers[i]->biases, gradient, layers[i]->nodeCount);

		delete[] gradient;

		Matrix* weightsTransposed = Matrix::transpose(layers[i]->weights);
		Matrix* errorMatrix = Matrix::multiply(weightsTransposed, errorMatrices[i], layers[i]->nodeCount);
		errorMatrices[i - 1] = errorMatrix->to1d();
		delete weightsTransposed;
		delete errorMatrix;
	}

	for (size_t i = 0; i < layerCount; ++i) {
		delete[] errorMatrices[i];
 	}
	delete[] errorMatrices;
}
