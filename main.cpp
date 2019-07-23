// main.cpp: Definiert den Einstiegspunkt für die Anwendung.
//

#include "main.h"
#include <core/network.h>
#include <core/layer.h>
#include <chrono>

void nnTest();

int main() {
	auto t1 = std::chrono::high_resolution_clock::now();
	nnTest();
	auto t2 = std::chrono::high_resolution_clock::now();

	auto executionTime = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
	std::cout << executionTime;
	std::cin.get();
	return 0;
}

void nnTest() {
	const int layercount = 4;
	Layer* layers[4] = {
		new Layer(2, SIGMOID),
		new Layer(3, SIGMOID),
		new Layer(4, SIGMOID),
		new Layer(1, SIGMOID)
	};


	Network* net = new Network(layers, layercount, 0.01, 0.9);

	double** inputs = new double* [4];
	double** outputs = new double* [4];
	for (size_t i = 0; i < 4; ++i) {
		inputs[i] = new double[2];
		outputs[i] = new double[1];
	}
	inputs[0][0] =	0;
	inputs[0][1] =	0;
	outputs[0][0] =	0;

	inputs[1][0] =	0;
	inputs[1][1] =	1;
	outputs[1][0] = 1;

	inputs[2][0] =	1;
	inputs[2][1] =	0;
	outputs[2][0] =	1;

	inputs[3][0] =	1;
	inputs[3][1] =	1;
	outputs[3][0] = 0;

	net->train(inputs, outputs, 4, 20000, 32, true);

	double input1[2] = { 0, 0 };
	double input2[2] = { 0, 1.0 };
	double input3[2] = { 1.0, 0 };
	double input4[2] = { 1.0, 1.0 };

	net->predict(input1, 2);
	net->printLastResult();
	net->predict(input2, 2);
	net->printLastResult();
	net->predict(input3, 2);
	net->printLastResult();
	net->predict(input4, 2);
	net->printLastResult();

	delete net;
}


