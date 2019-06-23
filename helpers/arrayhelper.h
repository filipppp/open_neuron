#ifndef arrayhelper_h
#define arrayhelper_h
#include "../layer.h"

class ArrayHelper {
public:
	static double* getArray(size_t rows, size_t cols, bool zero = false);

	static double* getZero1D(size_t length);
	static double* subtractArrays(const double* arr1, const double* arr2, size_t size);
	static double* mapTo(double* arr, size_t size, Activation func, bool derivative = false);
	static double* matrixTo1DArray(Matrix* singleDimensionMatrix);
	static double* multiply(double* x, double multiplier, size_t size);
	static void zero(double* arr, size_t length);
	static void add(double* x1, double* x2, size_t size);

	static double* hadamardArray(double* x1, double*x2, size_t size);

	static double maxNumber(const double* arr, size_t length);
};

#endif