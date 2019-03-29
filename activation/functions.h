#ifndef functions_h
#define functions_h

class Functions {
public:
	static double sigmoid(double x);
	static double dsigmoid(double x);

	static double identity(double x);
	static double didentity(double x);

	static double relu(double x);
	static double drelu(double x);
};


#endif