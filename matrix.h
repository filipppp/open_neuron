#ifndef matrix_h
#define matrix_h
#include "layer.h"

class Matrix {
	public:
		unsigned long rows;
		unsigned long cols;
		double** data;

		Matrix(unsigned long rows, unsigned long cols, bool init = false);
		Matrix(unsigned long rows, unsigned long cols, double** data);

		Matrix* random();
		Matrix* add(Matrix* input);
		Matrix* add(double* input, unsigned long length);
		double* apply(ActivationFunction type);
		double averageValue();
		void print();

		static Matrix* hadamard(Matrix* m1, Matrix* m2);
		static Matrix* subtract(Matrix* m1, Matrix* m2);
		static Matrix* transpose(Matrix* m1);
		static Matrix* copy(Matrix* toCopy);
		static Matrix* multiply(Matrix* m1, Matrix* m2);
		static Matrix* multiply(Matrix* m1, double* singleMatrix, unsigned long singleMatrixLength);
};

#endif