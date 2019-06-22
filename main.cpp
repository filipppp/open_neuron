// main.cpp: Definiert den Einstiegspunkt für die Anwendung.
//

#include "main.h"
#include "network.h"
#include "layer.h"
#include "../../../../../../Program Files (x86)/Microsoft Visual Studio/2019/Enterprise/VC/Tools/MSVC/14.21.27702/include/chrono"
typedef long long int64; typedef unsigned long long uint64;


void nnTest();

int main()
{
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


	Network* net = new Network(layers, layercount, 0.01, 20, 0.75);
	double input1[2] = { 0, 0 };
	double input2[2] = { 0, 1.0 };
	double input3[2] = { 1.0, 0 };
	double input4[2] = { 1.0, 1.0 };

	double output1[1] = { 0 };
	double output2[1] = { 1.0 };
	double output3[1] = { 1.0 };
	double output4[1] = { 0 };

	for (unsigned long i = 0; i < 200000; i++) {
		double random = (double)std::rand() / RAND_MAX;
		if (random < 0.25) {
			net->train(input1, 2, output1, 1);
		}
		else if (random > 0.25 && random < 0.5) {
			net->train(input2, 2, output2, 1);
		}
		else if (random > 0.5 && random < 0.75) {
			net->train(input3, 2, output3, 1);
		}
		else {
			net->train(input4, 2, output4, 1);
		}
	}
	net->predict(input3, 2);
	//net->printLastResult();

	delete net;
}


