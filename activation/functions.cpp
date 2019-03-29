#include "functions.h"
#include <cmath>

double Functions::sigmoid(double x) {
	return 1 / (1 + exp(-x));
}
double Functions::dsigmoid(double x) {
	return x * (1 - x);
}

double Functions::identity(double x) {
	return x;
}
double Functions::didentity(double x) {
	return 1;
}

double Functions::relu(double x) {
	return x > 0 ? x : 0;
}
double Functions::drelu(double x) {
	return x > 0 ? 1 : 0;
}