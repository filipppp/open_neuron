#ifndef layer_h
#define layer

class Layer;

class Network {
public:
	Layer** layers;
	double learningRate;
	unsigned short batchSize;
	double momentum;
	size_t layerCount;
	unsigned short batchTrained = 0;

	Network(Layer** layers, size_t layerCount, double learningRate, unsigned short batchSize, float momentum, bool init = false);
	~Network();

	double* predict(double* inputs, size_t length);
	void train(double* inputs, size_t inputLength, double* output, size_t outputLength);

	void printLastResult();
};

#endif
