// main.cpp: Definiert den Einstiegspunkt für die Anwendung.
//

#include "main.h"
#include "network.h"
#include "layer.h"


int main()
{

	const int layercount = 4;
	Layer* layers[layercount] = { 
		new Layer(2, SIGMOID),
		new Layer(4, SIGMOID),
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
		double random = (double) std::rand() / RAND_MAX;
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
	net->predict(input2, 2);
	net->printLastResult();

	std::cin.get();
	return 0;
}
