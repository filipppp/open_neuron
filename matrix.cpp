#include <cstdlib>
#include <stdexcept>
#include <iostream>
#include "matrix.h"
#include "helpers/arrayhelper.h"
#include "activation/functions.h"

/*
 *
	class RoworColumnNotMatchingException final : public std::exception {
	public:
		std::string errorMsg = "Rows and Columns don't match";
		char const* what() const override {
			return errorMsg.c_str();
		};
	};
 */

Matrix::Matrix(unsigned long rows, unsigned long cols, bool init) {
	this->rows = rows;
	this->cols = cols;
	this->data = ArrayHelper::getArray(rows, cols, init);
}

Matrix::Matrix(unsigned long rows, unsigned long cols, double **data) {
	this->rows = rows;
	this->cols = cols;
	this->data = ArrayHelper::getArray(rows, cols);
}

Matrix::~Matrix()
{
	for (int i = 0; i < rows; ++i)
	{
		delete[] data[i];
	}
	delete[] data;
}


Matrix* Matrix::random() {
	for (unsigned long i = 0; i < rows; i++) {
		for (unsigned long j = 0; j < cols; j++) {
			data[i][j] = ((double) std::rand() / RAND_MAX) * 2 - 1;
		}
	}
	return this;
}

Matrix* Matrix::add(Matrix* input) {
	if (input->rows != rows && input->cols != cols) { return nullptr; }
	for (unsigned long i = 0; i < rows; i++) {
		for (unsigned long j = 0; j < cols; j++) {
			data[i][j] += input->data[i][j];
		}
	}
	return this;
}

Matrix* Matrix::add(double* input, unsigned long length) {
	if (length != rows && length != cols) {
		std::cout << "Rows and cols don't match";
		return nullptr;
	}
	for (unsigned long i = 0; i < rows; i++) {
			data[i][0] += input[i];
	}
	return this;
}

double* Matrix::apply(Activation func) {
	if (cols != 1) { 
		std::cout << "Apply function only works on single Column Matrices";
		return nullptr; 
	}

	double* oneDimArray = ArrayHelper::matrixTo1DArray(this);
	return ArrayHelper::mapTo(oneDimArray, rows, func);
}

double* Matrix::to1d()
{
	if (cols != 1) { return nullptr; }

	double* output = new double[rows];
	for (unsigned long i = 0; i < rows; i++)
	{
		output[i] = data[i][0];
	}

	return output;
}

Matrix* Matrix::multiply(Matrix* m1, Matrix* m2) {
	if (m1->rows != m2->cols) { return nullptr; }

	Matrix* matrix = new Matrix(m1->rows, m2->cols);
	for (unsigned long i = 0; i < matrix->rows; i++) {
		for (unsigned long j = 0; j < matrix->cols; j++) {
			double sum = 0;
			for (unsigned long k = 0; k < m1->cols; k++) {
				sum += m1->data[i][k] * m2->data[k][j];
			}
			matrix->data[i][j] = sum;
		}
	}
	return matrix;
}

Matrix* Matrix::multiply(Matrix* m1, const double* singleMatrix, unsigned long singleMatrixLength) {
	if (m1->cols != singleMatrixLength)
	{
		std::cout << "Cols don't match rows";
		return nullptr;
	}

	Matrix* matrix = new Matrix(m1->rows, 1);
	for (unsigned long i = 0; i < matrix->rows; i++) {
		for (unsigned long j = 0; j < matrix->cols; j++) {
			double sum = 0;
			for (unsigned long k = 0; k < m1->cols; k++) {
				sum += m1->data[i][k] * singleMatrix[k];
			}
			matrix->data[i][j] = sum;
		}
	}
	return matrix;
}

Matrix* Matrix::multiply(double* x1, unsigned long x1Length, double* x2, unsigned long x2Length)
{
	if (1 != 1)
	{
		std::cout << "Cols don't match rows";
		return nullptr;
	}

	Matrix* matrix = new Matrix(x1Length, x2Length);
	for (unsigned long i = 0; i < x1Length; i++) {
		for (unsigned long j = 0; j < x2Length; j++) {
			double sum = 0;
			for (int k = 0; k < x2Length; ++k)
			{
				sum += x1[i] * x2[k];
			}
			matrix->data[i][j] = sum;
		}
	}
	return matrix;
}


double Matrix::averageValue() {
	double sum = 0;
	for (unsigned long i = 0; i < rows; i++) {
		for (unsigned long j = 0; j < cols; j++) {
			sum += data[i][j];
		}
	}
	return sum / (rows * cols);
}

Matrix* Matrix::print() {
	for (unsigned long i = 0; i < rows; i++) {
		for (unsigned long j = 0; j < cols; j++) {
			std::cout << data[i][j];
		}
		std::cout << std::endl;
	}
	return this;
}

Matrix* Matrix::hadamard(Matrix* m1, Matrix* m2) {
	if (m1->rows != m2->rows && m1->cols != m2->cols) { return nullptr; }

	Matrix* matrix = new Matrix(m1->rows, m1->cols);
	for (unsigned long i = 0; i < m1->rows; i++) {
		for (unsigned long j = 0; j < m1->cols; j++) {
			matrix->data[i][j] = m1->data[i][j] * m2->data[i][j];
		}
	}
	return matrix;
}

Matrix* Matrix::subtract(Matrix* m1, Matrix* m2) {
	if (m1->rows != m2->rows && m1->cols != m2->cols) { return nullptr; }

	Matrix* matrix = new Matrix(m1->rows, m1->cols);
	for (unsigned long i = 0; i < m1->rows; i++) {
		for (unsigned long j = 0; j < m1->cols; j++) {
			matrix->data[i][j] = m1->data[i][j] - m2->data[i][j];
		}
	}
	return matrix;
}

Matrix* Matrix::transpose(Matrix* m1) {
	Matrix* matrix = new Matrix(m1->cols, m1->rows);
	for (unsigned long i = 0; i < matrix->rows; i++) {
		for (unsigned long j = 0; j < matrix->cols; j++) {
			matrix->data[i][j] = m1->data[j][i];
		}
	}
	return matrix;
}

Matrix* Matrix::copy(Matrix* toCopy) {
	Matrix* matrix = new Matrix(toCopy->rows, toCopy->cols);
	for (unsigned long i = 0; i < matrix->rows; i++) {
		for (unsigned long j = 0; j < matrix->cols; j++) {
			matrix->data[i][j] = toCopy->data[i][j];
		}
	}
	return matrix;
}
