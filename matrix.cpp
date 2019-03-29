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


Matrix* Matrix::random() {
	for (unsigned long i = 0; i < rows; i++) {
		for (unsigned long j = 0; j < cols; j++) {
			data[i][j] = ((double) rand() / (RAND_MAX)) * 2 - 1;
		}
	}
	return this;
}

Matrix* Matrix::add(Matrix* input) {
	if (input->rows != rows && input->cols != cols) {
		return nullptr;
	}
	for (unsigned long i = 0; i < rows; i++) {
		for (unsigned long j = 0; j < cols; j++) {
			data[i][j] += input->data[i][j];
		}
	}
	return this;
}

Matrix* Matrix::add(double* input, unsigned long length) {
	if (length != rows && length != cols) {
		return nullptr;
	}
	for (unsigned long i = 0; i < rows; i++) {
			data[i][0] += input[i];
	}
	return this;
}

double* Matrix::apply(ActivationFunction type) {
	double* finalArray = new double[rows];
	for (unsigned long i = 0; i < rows; i++) {
		finalArray[i] = Functions::sigmoid(data[i][0]);
	}
	return finalArray;
}

Matrix* Matrix::multiply(Matrix* m1, Matrix* m2) {
	if (m1->rows != m2->cols) {
		return nullptr;
	}
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

Matrix* Matrix::multiply(Matrix* m1, double* singleMatrix, unsigned long singleMatrixLength) {
	if (m1->cols != singleMatrixLength) {
		return nullptr;
	}
	Matrix* matrix = new Matrix(m1->rows, singleMatrixLength);
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

double Matrix::averageValue() {
	double sum = 0;
	for (unsigned long i = 0; i < rows; i++) {
		for (unsigned long j = 0; j < cols; j++) {
			sum += data[i][j];
		}
	}
	return sum / (rows * cols);
}

void Matrix::print() {
	for (unsigned long i = 0; i < rows; i++) {
		for (unsigned long j = 0; j < cols; j++) {
			std::cout << data[i][j];
		}
		std::cout << std::endl;
	}
}

Matrix* Matrix::hadamard(Matrix* m1, Matrix* m2) {
	if (m1->rows != m2->rows && m1->cols != m2->cols) {
		return nullptr;
	}
	Matrix* matrix = new Matrix(m1->rows, m1->cols);
	for (unsigned long i = 0; i < m1->rows; i++) {
		for (unsigned long j = 0; j < m1->cols; j++) {
			matrix->data[i][j] = m1->data[i][j] * m2->data[i][j];
		}
	}
	return matrix;
}

Matrix* Matrix::subtract(Matrix* m1, Matrix* m2) {
	if (m1->rows != m2->rows && m1->cols != m2->cols) {
		return nullptr;
	}
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
