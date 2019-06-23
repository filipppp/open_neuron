#include "arrayhelper.h"
#include "../matrix.h"
#include <iostream>

double* ArrayHelper::getArray(size_t rows, size_t cols, bool zero) {
	double* array = new double[rows*cols];
	if (zero) {
		for (size_t i = 0; i < rows; i++) {
			for (size_t j = 0; j < cols; j++) {
				array[i * cols + j] = 0;
			}
		}
	} 
	return array;
}

double* ArrayHelper::getZero1D(size_t length) {
	double* arr = new double[length];
	for (size_t i = 0; i < length; i++) {
		arr[i] = 0;
	}
	return arr;
}

double* ArrayHelper::subtractArrays(const double* arr1, const double* arr2, size_t size) 
{
	double* result = new double[size];
	for (size_t i = 0; i < size; ++i)
	{
		result[i] = arr1[i] - arr2[i];
	}
	return result;
}

double* ArrayHelper::mapTo(double* arr, size_t size, Activation func, bool derivative)
{
	if (func == SOFTMAX)
	{
		return Functions::getMultiValue(arr, size, func, derivative);
	}

	double* output = new double[size];
	for (size_t i = 0; i < size; ++i)
	{
		output[i] = Functions::getValue(arr[i], func, derivative);
	}
	return output;
}

double* ArrayHelper::matrixTo1DArray(Matrix* m1)
{
	double* output = new double[m1->rows];
	for (size_t i = 0; i < m1->rows; ++i)
	{
		output[i] = m1->data[i * m1->cols];
	}
	return output;
}

double* ArrayHelper::multiply(double* x, double multiplier, size_t size)
{
	double* output = new double[size];
	for (size_t i = 0; i < size; i++)
	{
		output[i] = x[i] * multiplier;
	}
	return output;
}

void ArrayHelper::zero(double* arr, size_t length) {
	for (size_t i = 0; i < length; ++i) {
		arr[i] = 0;
	}
}

void ArrayHelper::add(double* x1, double* x2, size_t size)
{
	for (size_t i = 0; i < size; i++)
	{
		x1[i] += x2[i];
	}
}

double* ArrayHelper::hadamardArray(double* x1, double* x2, size_t size)
{
	double* output = new double[size];
	for (size_t i = 0; i < size; i++)
	{
		output[i] = x1[i] * x2[i];
	}
	return output;
}

double ArrayHelper::maxNumber(const double* arr, size_t length)
{
	double max = arr[0];
	for (size_t i = 1; i < length; i++)
	{
		if (arr[i] > max)
		{
			max = arr[i];
		}
	}
	return max;
}
