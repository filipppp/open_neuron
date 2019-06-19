#ifndef arrayhelper_h
#define arrayhelper_h
#include "../layer.h"

class ArrayHelper {
public:
	static double** getArray(unsigned long rows, unsigned long cols, bool zero = false);

	static double* getZero1D(unsigned long length);
	static double* subtractArrays(const double* arr1, const double* arr2, unsigned long size);
	static double* mapTo(double* arr, unsigned long size, Activation func, bool derivative = false);
	static double* matrixTo1DArray(Matrix* singleDimensionMatrix);
	static double* multiply(double* x, double multiplier, unsigned long size);
	static void add(double* x1, double* x2, unsigned long size);

	static double* hadamardArray(double* x1, double*x2, unsigned long size);

	static double maxNumber(const double* arr, unsigned long length);
};

#endif