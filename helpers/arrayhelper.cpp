#include "arrayhelper.h"

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
