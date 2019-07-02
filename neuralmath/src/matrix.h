#ifndef matrix_h
#define matrix_h
#include "functions.h"

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
		Matrix* divide(double quotient);
		Matrix* subtract(Matrix* m);
		Matrix* zero();

		double* apply(Activation func);
		double* to1d() const;
		double averageValue() const;

		static Matrix* hadamard(Matrix* m1, Matrix* m2);
		static Matrix* subtract(Matrix* m1, Matrix* m2);
		static Matrix* transpose(Matrix* m1);
		static Matrix* copy(Matrix* toCopy);
		static Matrix* multiply(Matrix* m1, Matrix* m2);
		static Matrix* multiply(Matrix* m1, const double* singleMatrix, size_t singleMatrixLength);
		static Matrix* multiply(double* x1, size_t x1Length, double* x2, size_t x2Length);

		static void moveData(Matrix* from, Matrix* to);
};

#endif