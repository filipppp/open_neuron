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

Matrix::Matrix(size_t rows, size_t cols, bool init) {
	this->rows = rows;
	this->cols = cols;
	this->data = ArrayHelper::getArray(rows, cols, init);
}

Matrix::Matrix(size_t rows, size_t cols, double* data) {
	this->rows = rows;
	this->cols = cols;
	this->data = ArrayHelper::getArray(rows, cols);
}

Matrix::~Matrix() {
	delete[] data;
}


Matrix* Matrix::random() {
	for (size_t i = 0; i < rows*cols; i++) {
		data[i] = ((double)std::rand() / RAND_MAX) * 2 - 1;
	}
	return this;
}

Matrix* Matrix::add(Matrix* input) {
	if (input->rows != rows || input->cols != cols) {
		std::cout << "Rows and Cols don't match up";
		return nullptr;
	}

	for (size_t i = 0; i < rows*cols; i++) {
		data[i] += input->data[i];
	}
	return this;
}

Matrix* Matrix::add(double* input, size_t length) {
	if (length != rows || cols != 1) {
		std::cout << "Rows and cols don't match";
		return nullptr;
	}

	for (size_t i = 0; i < rows; i++) {
			data[i * cols] += input[i];
	}
	return this;
}

double* Matrix::apply(Activation func) {
	if (cols != 1) { 
		std::cout << "Apply function only works on single Column Matrices";
		return nullptr; 
	}

	double* oneDimArray = to1d();
	ArrayHelper::mapTo(oneDimArray, rows, func);
	return oneDimArray;
}

double* Matrix::to1d() const {
	if (cols != 1) { return nullptr; }

	double* output = new double[rows];
	for (size_t i = 0; i < rows; i++) {
		output[i] = data[i * cols];
	}

	return output;
}

Matrix* Matrix::multiply(Matrix* m1, Matrix* m2) {
	if (m1->rows != m2->cols) {
		std::cout << "Parameter 1 ROWS don't match Parameter 2 COLS";
		return nullptr;
	}

	Matrix* matrix = new Matrix(m1->rows, m2->cols);
	for (size_t i = 0; i < matrix->rows; i++) {
		for (size_t j = 0; j < matrix->cols; j++) {
			double sum = 0;
			for (size_t k = 0; k < m1->cols; k++) {
				sum += m1->data[i * m1->cols + k] * m2->data[k * m2->cols + j];
			}
			matrix->data[i * matrix->cols + j] = sum;
		}
	}
	return matrix;
}

Matrix* Matrix::multiply(Matrix* m1, const double* singleMatrix, size_t singleMatrixLength) {
	if (m1->cols != singleMatrixLength) {
		std::cout << "Cols don't match rows";
		return nullptr;
	}

	Matrix* matrix = new Matrix(m1->rows, 1);
	for (size_t i = 0; i < matrix->rows; i++) {
		for (size_t j = 0; j < matrix->cols; j++) {
			double sum = 0;
			for (size_t k = 0; k < m1->cols; k++) {
				sum += m1->data[i * m1->cols + k] * singleMatrix[k];
			}
			matrix->data[i * matrix->cols + j] = sum;
		}
	}
	return matrix;
}

Matrix* Matrix::multiply(double* x1, size_t x1Length, double* x2, size_t x2Length) {
	if (1 != 1) {
		std::cout << "Cols don't match rows";
		return nullptr;
	}

	Matrix* matrix = new Matrix(x1Length, x2Length);
	for (size_t i = 0; i < x1Length; i++) {
		for (size_t j = 0; j < x2Length; j++) {
			double sum = 0;
			for (size_t k = 0; k < x2Length; ++k) {
				sum += x1[i] * x2[k];
			}
			matrix->data[i * x2Length + j] = sum;
		}
	}
	return matrix;
}

Matrix* Matrix::multiply(double multiplier) {
	for (size_t i = 0; i < rows*cols; i++) {
			data[i] *= multiplier;
	}
	return this;
}

Matrix* Matrix::divide(double quotient) {
	for (size_t i = 0; i < rows * cols; i++) {
		data[i] /= quotient;
	}
	return this;
}

Matrix* Matrix::subtract(Matrix* m) {
	if (rows != m->rows || cols != m->cols) {
		std::cout << "Rows and Columns of both matrices don't match";
		return nullptr;
	}

	for (size_t i = 0; i < rows * cols; i++) {
		data[i] -= m->data[i];
	}
	return this;
}

Matrix* Matrix::zero() {
	for (size_t i = 0; i < rows*cols; i++) {
			data[i] = 0;
	}
	return this;
}


double Matrix::averageValue() const {
	double sum = 0;
	for (size_t i = 0; i < rows*cols; i++) {
			sum += data[i];
	}
	return sum / (rows * cols);
}

Matrix* Matrix::print() {
	for (size_t i = 0; i < rows; i++) {
		for (size_t j = 0; j < cols; j++) {
			std::cout << data[i * cols + j];
		}
		std::cout << std::endl;
	}
	return this;
}

Matrix* Matrix::hadamard(Matrix* m1, Matrix* m2) {
	if (m1->rows != m2->rows || m1->cols != m2->cols) {
		std::cout << "Rows and Columns of both matrices don't match";
		return nullptr;
	}

	Matrix* matrix = new Matrix(m1->rows, m1->cols);
	for (size_t i = 0; i < matrix->rows*matrix->cols; i++) {
			matrix->data[i] = m1->data[i] * m2->data[i];
	}
	return matrix;
}

Matrix* Matrix::subtract(Matrix* m1, Matrix* m2) {
	if (m1->rows != m2->rows || m1->cols != m2->cols) {
		std::cout << "Rows and Columns of both matrices don't match";
		return nullptr;
	}

	Matrix* matrix = new Matrix(m1->rows, m1->cols);
	for (size_t i = 0; i < matrix->rows*matrix->cols; i++) {
			matrix->data[i] = m1->data[i] - m2->data[i];
	}
	return matrix;
}

Matrix* Matrix::transpose(Matrix* m1) {
	Matrix* matrix = new Matrix(m1->cols, m1->rows);
	for (size_t i = 0; i < matrix->rows; i++) {
		for (size_t j = 0; j < matrix->cols; j++) {
			matrix->data[i * matrix->cols + j] = m1->data[j * m1->cols + i];
		}
	}
	return matrix;
}

Matrix* Matrix::copy(Matrix* toCopy) {
	Matrix* matrix = new Matrix(toCopy->rows, toCopy->cols);
	for (size_t i = 0; i < matrix->rows*matrix->cols; i++) {
			matrix->data[i] = toCopy->data[i];
	}
	return matrix;
}

void Matrix::moveData(Matrix* from, Matrix* to) {
	if (from->rows != to->rows || from->cols != to->cols) {
		std::cout << "Can't move data because the rows and columns don't match up";
		return;
	}

	for (size_t i = 0; i < from->rows * from->cols; i++) {
		to->data[i] = from->data[i];
	}
}
