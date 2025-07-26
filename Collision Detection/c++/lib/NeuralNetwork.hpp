#ifndef NEURALNETWORK_HPP
#define NEURALNETWORK_HPP

#include <vector>
#include <stdexcept>
#include <cmath>
#include <chrono>
#include "Layer.hpp"
#include "Matrix.hpp"

class NeuralNetwork {
    private:
        std::vector<Layer> layers;
        std::vector<Matrix> weights;

        void connect_layers() {
            for (size_t i = 0; i < layers.size() - 1; ++i) {
                // Setup the weights with random values
                weights.emplace_back(layers[i].size, layers[i + 1].size);
            }
        }

        void initializeWeights() {
            for (auto& ws: weights) {
                ws.fillRandom();

                ws *= sqrt(2.0 / ws.getRows()); // Good for ReLU activation functions
            }
        }

    public:
        NeuralNetwork(const std::vector<Layer>& network_layers) : layers(network_layers) {
            connect_layers();
            initializeWeights();
        }

        std::vector<double> forward(const std::vector<double>& input) {
            if (input.size() != layers[0].size) {
                throw std::runtime_error("Input size mismatch");
            }

            layers[0].a = input; // Activation output values

            // Weights connect layers -> could also use weights.size() and not layers.size() - 1
            // We're using layers.size() - 1 for readability and clarity because we're thinking in terms of layers and not weights
            for (size_t l = 0; l < layers.size() - 1; ++l) {
                // Calculate the dot product
                for (size_t j = 0; j < weights[l].getCols(); ++j) {
                    double sum = layers[l + 1].bias[j];

                    for (size_t i = 0; i < weights[l].getRows(); ++i) {
                        sum += layers[l].a[i] * weights[l](i,j); // Weighted sum
                    }

                    // Store pre activation value and activated output for each neuron in next layer
                    layers[l + 1].z[j] = sum; // Pre activation value
                    layers[l + 1].a[j] = layers[l + 1].applyActivation(sum); // Activated output
                }
            }

            return layers.back().a; // Return activation of last layer (output layer -> prediction of the network)
        }

        // Predict method
        std::vector<double> predict(const std::vector<double>& input) {
            return forward(input);
        }
};

#endif