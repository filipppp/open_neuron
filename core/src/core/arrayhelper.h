#ifndef arrayhelper_h
#define arrayhelper_h
#include "layer.h"
#include "functions.h"


class ArrayHelper {
public:
	static double* getArray(size_t rows, size_t cols, bool zero = false);

	static double* getZero1D(size_t length);
	static double* subtractArrays(const double* arr1, const double* arr2, size_t size, double* memory);
	static double* hadamardArray(double* x, double* x2, size_t size);

	static void zero(double* arr, size_t length);
	static void mapTo(double* arr, size_t size, Activation func, bool derivative = false);
	static void add(double* x1, double* x2, size_t size);
	static void divide(double* x, double qoutient, size_t size);
	static void multiply(double* x, double multiplier, size_t size);
	static void print(double* x, size_t size);
	static void subtract(double* x1, double* x2, size_t size);
	static void copy(double* from, double* to, size_t size);
	static void shuffleTrainingData(double** inputs, double** outputs, size_t trainingCount);

	static double averageValue(double* x, size_t size);
	static double maxNumber(const double* arr, size_t length);
};

#endif
