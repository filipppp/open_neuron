#ifndef layer_h
#define layer

class Layer;

class Network {
public:
	Layer** layers;
	size_t layerCount;

	double learningRate;
	double momentum;

	unsigned short batchTrained = 0;
	unsigned short batchSize;

	Network(Layer** layers, size_t layerCount, double learningRate, unsigned short batchSize, float momentum, bool init = false);
	~Network();

	double* predict(double* inputs, size_t length) const;
	void train(double* inputs, size_t inputLength, double* output, size_t outputLength);

	void printLastResult();
};

#endif
