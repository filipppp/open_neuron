# OpenNeuron ☜(ﾟヮﾟ☜)

[![open_neuron](http://kekomat11.ddnss.de/n.png)](https://openneuron.cf)


OpenNeuron is a C++ Neural Network which implements the following features:

  - Customizable Models 
  - Beginner friendly API
  - Different Activation and Loss Functions
  - Lots of optimizers!

### Installation

There are two ways to use OpenNeuron:

##### Build

To build from the source yourself, clone this repo 

###### Linux

Run the build script provided in the repo, the build will appear in the dist/ folder
```sh
$ ./build.sh
```
###### Windows

Use the CMake GUI

##### Source
If you want to use the source code itself

```sh
$ git clone https://github.com/filipppp/open_neuron.git
$ cd ./open_neuron
```

Include the *network.h* file to start and to create your network

```sh
#include <core/network.h>

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
 - Implement RNNs
 - Implement LSTMs
 
 - Implement multiple optimizers
 - Implement multiple loss functions

License
----

MIT


