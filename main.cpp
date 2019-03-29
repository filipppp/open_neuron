// main.cpp: Definiert den Einstiegspunkt für die Anwendung.
//

#include "main.h"
#include "network.h"
#include "layer.h"


int main()
{

	const int layercount = 4;
	Layer* layers[layercount] = { new Layer(2), new Layer(4), new Layer(4), new Layer(1)};

	Network* net = new Network(layers, layercount, 0.01, 20, 0.75);
	double inputs[2] = {0, 1.0};
	net->predict(inputs, 2);


	std::cin.get();
	return 0;
}
