#include "functions.h"
#include <cmath>
#include "../helpers/arrayhelper.h"
#include <iostream>
#include <cmath>

/* Single Parameter Activation Functions */


double Functions::sigmoid(double x)
{
	return 1 / (1 + exp(-x));
}

double Functions::derivativeSigmoid(double x)
{
	return x * (1 - x);
}

double Functions::relu(double x)
{
	return x > 0 ? x : 0;
}

double Functions::derivativeRelu(double x)
{
	return x > 0 ? 1 : 0;
}

double Functions::identity(double x)
{
	return x;
}

double Functions::derivativeIdentity(double x)
{
	return 1;
}

double Functions::binary(double x)
{
	return x >= 0 ? 1 : 0;
}

double Functions::derivativeBinary(double x)
{
	return 0;
}

double Functions::tanh(double x)
{
	return (exp(x) - exp(-x)) / (exp(x) + exp(-x));
}

double Functions::derivativeTanh(double x)
{
	return 1 - pow(tanh(x), 2);
}

/* Multi Parameter Functions */

double* Functions::softmax(double* arr, unsigned long size)
{
	double* outputs = new double[size];
	double sum = 0;
	double max = ArrayHelper::maxNumber(arr, size);
	for (unsigned long i = 0; i < size; i++)
	{
		/* number - max to decrease size of numbers because of exponential nature */
		sum += exp(arr[i] - max);
	}
	for (unsigned long i = 0; i < size; i++)
	{
		outputs[i] = exp(arr[i] - max) / sum;
	}
	return outputs;
}

double* Functions::derivativeSoftmax(double* arr, unsigned long size)
{
	/* iwann mal */
	return nullptr;
}

double Functions::getValue(double x, Activation func, bool derivative)
{
	switch (func)
	{
	case SIGMOID:
		return derivative ? derivativeSigmoid(x) : sigmoid(x);
	case RELU:
		return derivative ? derivativeRelu(x) : relu(x);
	case IDENTITY:
		return derivative ? derivativeIdentity(x) : identity(x);
	case BINARY:
		return derivative ? derivativeBinary(x) : binary(x);
	case TANH:
		return derivative ? derivativeTanh(x) : tanh(x);
	default:
		return derivative ? derivativeSigmoid(x) : sigmoid(x);
	}
}

double* Functions::getMultiValue(double* arr, unsigned long size, Activation func, bool derivative)
{
	switch (func)
	{
	case SOFTMAX:
		return derivative ? derivativeSoftmax(arr, size) : softmax(arr, size);
	}
}
