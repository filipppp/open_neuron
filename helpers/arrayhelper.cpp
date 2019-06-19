#include "arrayhelper.h"
#include "../matrix.h"
#include <iostream>

double** ArrayHelper::getArray(unsigned long rows, unsigned long cols, bool zero) {
	double** array = new double*[rows];
	if (zero) {
		for (unsigned long i = 0; i < rows; i++) {
			array[i] = new double[cols];
			for (unsigned long j = 0; j < cols; j++) {
				array[i][j] = 0;
			}
		}
	} else {
		for (unsigned long i = 0; i < rows; i++) {
			array[i] = new double[cols];
		}
	}
	return array;
}

double* ArrayHelper::getZero1D(unsigned long length) {
	double* arr = new double[length];
	for (unsigned long i = 0; i < length; i++) {
		arr[i] = 0;
	}
	return arr;
}

double* ArrayHelper::subtractArrays(const double* arr1, const double* arr2, unsigned long size) 
{
	double* result = new double[size];
	for (unsigned long i = 0; i < size; ++i)
	{
		result[i] = arr1[i] - arr2[i];
	}
	return result;
}

double* ArrayHelper::mapTo(double* arr, unsigned long size, Activation func, bool derivative)
{
	if (func == SOFTMAX)
	{
		return Functions::getMultiValue(arr, size, func, derivative);
	}

	double* output = new double[size];
	for (unsigned long i = 0; i < size; ++i)
	{
		output[i] = Functions::getValue(arr[i], func, derivative);
	}
	return output;
}

double* ArrayHelper::matrixTo1DArray(Matrix* m1)
{
	double* output = new double[m1->rows];
	for (unsigned long i = 0; i < m1->rows; ++i)
	{
		output[i] = m1->data[i][0];
	}
	return output;
}

double* ArrayHelper::multiply(double* x, double multiplier, unsigned long size)
{
	double* output = new double[size];
	for (unsigned long i = 0; i < size; i++)
	{
		output[i] = x[i] * multiplier;
	}
	return output;
}

void ArrayHelper::add(double* x1, double* x2, unsigned long size)
{
	for (unsigned long i = 0; i < size; i++)
	{
		x1[i] += x2[i];
	}
}

double* ArrayHelper::hadamardArray(double* x1, double* x2, unsigned long size)
{
	double* output = new double[size];
	for (unsigned long i = 0; i < size; i++)
	{
		output[i] = x1[i] * x2[i];
	}
	return output;
}

double ArrayHelper::maxNumber(const double* arr, unsigned long length)
{
	double max = arr[0];
	for (unsigned long i = 1; i < length; i++)
	{
		if (arr[i] > max)
		{
			max = arr[i];
		}
	}
	return max;
}
