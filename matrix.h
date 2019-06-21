#ifndef matrix_h
#define matrix_h
#include "activation/functions.h"

class Matrix {
	public:
		unsigned long rows;
		unsigned long cols;
		double** data;

		Matrix(unsigned long rows, unsigned long cols, bool init = false);
		Matrix(unsigned long rows, unsigned long cols, double** data);
		~Matrix();

		Matrix* random();
		Matrix* add(Matrix* input);
		Matrix* print();
		Matrix* add(double* input, unsigned long length);

		double* apply(Activation func);
		double* to1d();
		double averageValue();

		static Matrix* hadamard(Matrix* m1, Matrix* m2);
		static Matrix* subtract(Matrix* m1, Matrix* m2);
		static Matrix* transpose(Matrix* m1);
		static Matrix* copy(Matrix* toCopy);
		static Matrix* multiply(Matrix* m1, Matrix* m2);
		static Matrix* multiply(Matrix* m1, const double* singleMatrix, unsigned long singleMatrixLength);
		static Matrix* multiply(double* x1, unsigned long x1Length, double* x2, unsigned long x2Length);
};

#endif