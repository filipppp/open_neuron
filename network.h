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
	Network(Layer** layers, size_t layerCount, double learningRate, float momentum, bool init = false);
	~Network();

	double* predict(double* inputs, size_t length) const;
	void train(double** inputs, double** outputs, size_t trainingSize, size_t epochs = 2, size_t batchSize = 20, bool verbose = false);
	void applyMiniBatch();

	void printLastResult();
};

#endif
