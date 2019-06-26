#ifndef functions_h
#define functions_h


enum Activation { SIGMOID, RELU, IDENTITY, BINARY, TANH, SOFTMAX};

class Functions
{
public:
	static double sigmoid(double x);
	static double derivativeSigmoid(double x);

	static double relu(double x);
	static double derivativeRelu(double x);

	static double identity(double x);
	static double derivativeIdentity(double x);

	static double binary(double x);
	static double derivativeBinary(double x);

	static double tanh(double x);
	static double derivativeTanh(double x);

	static void softmax(double* arr, size_t size);
	static void derivativeSoftmax(double* arr, size_t size);

	static double getValue(double x, Activation func, bool derivative = false);
	static void getMultiValue(double* arr, size_t size, Activation func, bool derivative = false);
};

#endif