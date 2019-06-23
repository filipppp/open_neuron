#ifndef matrix_h
#define matrix_h
#include "activation/functions.h"

class Matrix {
	public:
		size_t rows;
		size_t cols;
		double* data;

		Matrix(size_t rows, size_t cols, bool init = false);
		Matrix(size_t rows, size_t cols, double* data);
		~Matrix();

		Matrix* random();
		Matrix* add(Matrix* input);
		Matrix* print();
		Matrix* add(double* input, size_t length);
		Matrix* multiply(double multiplier);
		Matrix* zero();

		double* apply(Activation func);
		double* to1d();
		double averageValue() const;

		static Matrix* hadamard(Matrix* m1, Matrix* m2);
		static Matrix* subtract(Matrix* m1, Matrix* m2);
		static Matrix* transpose(Matrix* m1);
		static Matrix* copy(Matrix* toCopy);
		static Matrix* multiply(Matrix* m1, Matrix* m2);
		static Matrix* multiply(Matrix* m1, const double* singleMatrix, size_t singleMatrixLength);
		static Matrix* multiply(double* x1, size_t x1Length, double* x2, size_t x2Length);
};

#endif