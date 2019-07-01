#include "network.h"
#include <array>
#include "layer.h"
#include <arrayhelper.h>
#include "matrix.h"
#include <ctime>
#include <iostream>

Network::Network(Layer** layers, size_t layerCount, double learningRate, float momentum,
	bool init) {
	srand(static_cast<unsigned int>(time(NULL)));

	this->layers = layers;
	this->learningRate = learningRate;
	this->momentum = momentum;
	this->layerCount = layerCount;

	for (size_t i = 1; i < layerCount; i++) {
		layers[i]->initWeights(layers[i - 1]);
	}
}

Network::~Network() {
	for (size_t i = 0; i < layerCount; ++i) {
		delete layers[i];
	}
}

double* Network::predict(double* inputs, size_t length) const {
	if (length != layers[0]->nodeCount) {
		std::cout << "Inputs don't match Input Model Layer";
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

void Network::train(double** inputs, double** outputs, size_t trainingSize, size_t epochs, size_t batchSize, bool verbose) {

	for (size_t epochCount = 0; epochCount < epochs; epochCount++) {
		ArrayHelper::shuffleTrainingData(inputs, outputs, trainingSize);


		for (size_t trainingCount = 0; trainingCount < trainingSize; trainingCount++) {
			/* Calculate Error which Network gives for specific input */
			double* predictedOutput = predict(inputs[trainingCount], layers[0]->nodeCount);
			double* outputError = ArrayHelper::subtractArrays(outputs[trainingCount], predictedOutput, layers[layerCount-1]->nodeCount);

			double** errorMatrices = new double* [layerCount];
			errorMatrices[layerCount - 1] = outputError;

			/* Iterating through layers from behind */
			for (size_t i = layerCount - 1; i > 0; i--) {
				/* Calculate neurons derivative */
				ArrayHelper::mapTo(layers[i]->neurons, layers[i]->nodeCount, layers[i]->func, true);

				/* Calculate gradient with formula: activation'(neurons) ⊗ Error * learningRate x input */
				double* gradientBase = ArrayHelper::hadamardArray(layers[i]->neurons, errorMatrices[i], layers[i]->nodeCount);
				/*
				 * Calculate how to adjust weights of current layer
				 * Adjust sum of batch (will be added afterwards to the real weights of the network)
				 */
				Matrix* deltaWeights = Matrix::multiply(gradientBase, layers[i]->nodeCount, layers[i - 1]->neurons, layers[i - 1]->nodeCount);
				layers[i]->deltaWeightSum->add(deltaWeights);
				delete deltaWeights;

				/* Adjust bias sum */
				ArrayHelper::add(layers[i]->deltaBiasSum, gradientBase, layers[i]->nodeCount);

				delete[] gradientBase;

				/* Calculate error output for previous layer for further calculation */
				Matrix* weightsTransposed = Matrix::transpose(layers[i]->weights);
				Matrix* errorMatrix = Matrix::multiply(weightsTransposed, errorMatrices[i], layers[i]->nodeCount);
				errorMatrices[i - 1] = errorMatrix->to1d();

				delete weightsTransposed;
				delete errorMatrix;
			}
			batchTrained++;

			/* Check if batch training is finished */
			if (batchTrained >= batchSize) {
				applyMiniBatch();

				/* Debug */
				if (verbose) {
					double average = 0;
					for (size_t i = 0; i < layers[layerCount - 1]->nodeCount; i++) {
						average += pow(errorMatrices[layerCount - 1][i], 2);
					}
					std::cout << "Batch Error: " << average / layers[layerCount - 1]->nodeCount << std::endl;
				}
			}

			/* Clear Error Matrices */
			for (size_t i = 0; i < layerCount; i++) {
				delete[] errorMatrices[i];
			}
			delete[] errorMatrices;
		}
	}
}

void Network::applyMiniBatch() {
		for (size_t i = layerCount - 1; i > 0; i--) {
			/* Mutliply Learning Rate + momentum with Formula
			 * delta = previousDelta * momentum + learningRate * ∇Loss
			 */
			layers[i]->deltaWeightSum
				->multiply(learningRate)
				->add(layers[i]->previousDeltaWeight->multiply(momentum));

			ArrayHelper::multiply(layers[i]->deltaBiasSum, learningRate, layers[i]->nodeCount);
			ArrayHelper::multiply(layers[i]->previousDeltaBias, momentum, layers[i]->nodeCount);
			ArrayHelper::add(layers[i]->deltaBiasSum, layers[i]->previousDeltaBias, layers[i]->nodeCount);

			/* Delete last previous deltas and set the new one (momentum optimization) */
			Matrix::moveData(layers[i]->deltaWeightSum, layers[i]->previousDeltaWeight);
			ArrayHelper::copy(layers[i]->deltaBiasSum, layers[i]->previousDeltaBias, layers[i]->nodeCount);

			/* Adjust calculated batch data for weights and biases */
			layers[i]->weights->add(layers[i]->deltaWeightSum);
			ArrayHelper::add(layers[i]->biases, layers[i]->deltaBiasSum, layers[i]->nodeCount);

			/* Set calculated batch to zero again*/
			layers[i]->deltaWeightSum->zero();
			ArrayHelper::zero(layers[i]->deltaBiasSum, layers[i]->nodeCount);
		}
		batchTrained = 0;
}

void Network::printLastResult() {
	for (size_t i = 0; i < layers[layerCount - 1]->nodeCount; i++) {
		std::cout << "Prediction: " << layers[layerCount - 1]->neurons[i] << std::endl;
	}
}