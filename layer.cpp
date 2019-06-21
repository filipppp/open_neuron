#include "layer.h"
#include "helpers/arrayhelper.h"
#include "activation/functions.h"
#include "../../../../../../Program Files (x86)/Microsoft Visual Studio/2019/Enterprise/VC/Tools/MSVC/14.21.27702/include/cstdlib"
#include <iostream>

Layer::Layer(unsigned long nodeCount, Activation func) {
	neurons = ArrayHelper::getZero1D(nodeCount);
	biases = ArrayHelper::getZero1D(nodeCount);
	deltaBiasSum = ArrayHelper::getZero1D(nodeCount);
	this->nodeCount = nodeCount;
	this->func = func;
}

Layer::~Layer()
{
	try
	{

		delete[] biases;
		std::cout << neurons[0];
		delete[] neurons;

		delete weights;
		delete deltaWeightSum;
		delete deltaBiasSum;
	}
	catch (const std::exception& ex)
	{
		std::cout << ex.what();
	}
}

