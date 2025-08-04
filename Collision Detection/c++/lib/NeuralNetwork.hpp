#ifndef NEURALNETWORK_HPP
#define NEURALNETWORK_HPP

#include <vector>
#include <stdexcept>
#include <cmath>
#include <chrono>
#include "Layer.hpp"
#include "Matrix.hpp"
#include "Log.hpp"
#include <sstream>

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

        // (Inputs -> Data used to train our model,
        // Targets -> Actual values / labels that should be the output,
        // Learning Rate -> How much we adjust the weights and biases -> the smaller the rate the slower the learning but also the more stable it is
        // Epochs -> How many times we gonna train the model
        // Batch Size -> Instead of adjusting the weights after every single input, we do it in batches (After X amount of inputs - mini batch training)
        // Verbose -> logging of the progress and process)
        void train(
            const std::vector<std::vector<double>>& inputs,const std::vector<std::vector<double>>& targets,
            double learning_rate, size_t epochs, size_t batch_size = 1, bool verbose = true
        ) {
            if (inputs.size() != targets.size()) {
                throw std::runtime_error("Input and Target sizes do not match");
            }

            // Number of learning rate, epochs and batch size must be above 0, if any of them are 0 or less, we throw an error
            if (learning_rate <= 0.0 || epochs <= 0.0 || batch_size <= 0.0) {
                throw std::runtime_error("Learning rate, epochs and batch size must be positive");
            }

            size_t dataset_size = inputs.size(); // Will make it easier to read later on
            auto start = std::chrono::high_resolution_clock::now(); // Time our training
            double totalError = 0.0; // Will accumulate the loss (Will be printed every 100 epochs)

            // At the begining of training, a high learning rate will make bigger progress but as you progress, you'll need to use a lower learning rate for the data to be stable
            // This strategy is called learning rate decay
            double base_lr = learning_rate; // Starting learning rate
            double decay_rate = 0.996; // 0.04% per epoch
            double min_lr = 1e-4; // 10^-4 -> 0.0001

            if (verbose) {
                // Write log to stringstream
                std::stringstream data;
                data << "############### Training Info ###############\n";
                data << "Learning rate: " << learning_rate
                    << "\t Learning rate decay rate: " << (1.0 - decay_rate) * 100 << "%"
                    << "\t EPOCHS: " << epochs << "\n";

                data << "Dataset size: " << dataset_size
                    << "\t Batch size: " << batch_size << "\n";
                data << "\n-------------- training.... --------------n";

                // Write to console
                std::cout << data.str();
                // Write to file
                L::log(data.str());
            }

            for (int epoch = 0; epoch < epochs; ++epoch) {
                learning_rate = std::max(min_lr, base_lr * std::powf(decay_rate, epoch)); // Had to use powf instead of pow
                // TODO   
            }

        }
};

#endif