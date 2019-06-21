#ifndef layer_h
#define layer

class Layer;

class Network {
public:
	Layer** layers;
	double learningRate;
	double batchSize;
	double momentum;
	unsigned long layerCount;
	unsigned long batchTrained = 0;

	Network(Layer** layers, unsigned long layerCount, double learningRate, unsigned short batchSize, float momentum, bool init = false);
	~Network();

	double* predict(double* inputs, unsigned long length);
	void train(double* inputs, unsigned long inputLength, double* output, unsigned long outputLength);

	void printLastResult();
};

#endif
