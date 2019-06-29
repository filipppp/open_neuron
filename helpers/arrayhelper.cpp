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

double* ArrayHelper::subtractArrays(const double* arr1, const double* arr2, size_t size) {
	double* result = new double[size];
	for (size_t i = 0; i < size; ++i) {
		result[i] = arr1[i] - arr2[i];
	}
	return result;
}

void ArrayHelper::mapTo(double* arr, size_t size, Activation func, bool derivative) {
	if (func == SOFTMAX) {
		Functions::getMultiValue(arr, size, func, derivative);
	}

	for (size_t i = 0; i < size; ++i) {
		arr[i] = Functions::getValue(arr[i], func, derivative);
	}
}

void ArrayHelper::multiply(double* x, double multiplier, size_t size)
{
	for (size_t i = 0; i < size; i++) {
		x[i] *= multiplier;
	}
}

void ArrayHelper::zero(double* arr, size_t length) {
	for (size_t i = 0; i < length; ++i) {
		arr[i] = 0;
	}
}

void ArrayHelper::add(double* x1, double* x2, size_t size) {
	for (size_t i = 0; i < size; i++) {
		x1[i] += x2[i];
	}
}

void ArrayHelper::divide(double* x, double qoutient, size_t size) {
	for (size_t i = 0; i < size; ++i) {
		x[i] /= qoutient;
	}
}

void ArrayHelper::print(double* x, size_t size) {
	for (size_t i = 0; i < size; i++) {
		std::cout << x[i] << " , ";
	}
	std::cout << std::endl;
}

void ArrayHelper::subtract(double* x1, double* x2, size_t size) {
	for (size_t i = 0; i < size; ++i) {
		x1[i] -= x2[i];
	}
}

void ArrayHelper::copy(double* from, double* to, size_t size) {
	for (size_t i = 0; i < size; i++) {
		to[i] = from[i];
	}
}

double ArrayHelper::averageValue(double* x, size_t size) {
	double sum = 0;
	for (size_t i = 0; i < size; i++) {
		sum += abs(x[i]);
	}
	return sum / size;
}

double* ArrayHelper::hadamardArray(double* x1, double* x2, size_t size) {
	double* output = new double[size];
	for (size_t i = 0; i < size; i++) {
		output[i] = x1[i] * x2[i];
	}
	return output;
}

double ArrayHelper::maxNumber(const double* arr, size_t length) {
	double max = arr[0];
	for (size_t i = 1; i < length; i++) {
		if (arr[i] > max) {
			max = arr[i];
		}
	}
	return max;
}
