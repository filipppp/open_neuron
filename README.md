# OpenNeuron

[![N|Solid](https://cldup.com/dTxpPi9lDf.thumb.png)](https://nodesource.com/products/nsolid)


OpenNeuron is a C++ Neural Network which implements the following features:

  - Customizable Models
  - Beginner friendly API
  - Different Activation and Loss Functions
  - Lots of optimizers!

### Installation

There are two ways to use OpenNeuron:

##### CLI

To use the CLI download the build for your OS
```sh
$ ./open-neuron
```
##### Source
If you want to use the source code itself

```sh
$ git clone https://github.com/filipppp/open_neuron.git
$ cd ./open_neuron
```

Include the *network.h* file to start and to create your network

```sh
#include "open_neuron/network.h"

void main() {
    Network* net = new Network(layers, layerCount, learningRate, momentum);
    net->predict();
    net->train();
}
```



### Development

Want to contribute? Great!

If you want to add any features, just fork this repository and make a pull request when you are done.



### Todos

 - Implement CNNs
 - Implement multiple optimizers
 - Implement multiple loss functions

License
----

MIT


