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

            for (int epoch = 0; epoch < epochs; ++epoch) { // training session iterator
                learning_rate = std::max(min_lr, base_lr * std::powf(decay_rate, epoch)); // Had to use powf instead of pow
                totalError = 0.0;
                
                for (size_t batch = 0; batch < dataset_size; batch += batch_size) { // mini batch training
                    size_t actual_batch_size = std::min(batch_size, (dataset_size - batch)); // Handles the case when the remaining dataset contains less elements than the batch size

                    std::vector<Matrix> weight_batch_gradients;
                    for (size_t i = 0; i < weights.size(); ++i) { // Create a gradient matrix
                        weight_batch_gradients.emplace_back(weights[i].getRows(), weights[i].getCols()); // Will hold the sum of gradients for each mini batch
                    }

                    std::vector<std::vector<double>> bias_batch_gradients(layers.size() - 1); // Create one vector for each layer (excluding the input layer itself)

                    // Resize each vector to match the number of neurons in its corresponding layer
                    for (size_t i = 0; i < bias_batch_gradients.size(); ++i) {
                        bias_batch_gradients[i].resize(layers[i + 1].size, 0.0);
                    }

                    // Loop through each training example in the current batch
                    for (size_t k = batch; k < (batch + actual_batch_size); ++k) {
                        forward(inputs[k]);

                        Layer& outputLayer = layers.back();
                        // Log(0) is undefined in math, in practice it is negative infinity (Completely breaks our loss function)
                        const double epsilon = 1e-7; // Used to prevent Log(0) -> Undefined

                        // Collision -> 1, no collision -> 0
                        for (size_t i = 0; i < outputLayer.size; ++i) {
                            double y_true = targets[k][i]; // Actual value that needs to be the result
                            double y_pred = outputLayer.a[i]; // Predicted value by the network

                            y_pred = std::min(std::max(y_pred, epsilon), 1.0 - epsilon);

                            totalError += (y_true * std::log(y_pred) + (1.0 - y_true) * std::log(1.0 - y_pred)); // Calculate loss

                            outputLayer.gradient[i] = y_pred - y_true; // Gradient for the output neauron
                        }

                        for (int l = (static_cast<int>(layers.size()) - 2); l > 0; --l) { // Go over all the layers except the output layer and last hidden layer
                            for (size_t i = 0; i < weights[l].getRows(); ++i) {
                                double error = 0.0; // Total weighted error from the next layer
                                for (size_t j = 0; weights[l].getCols(); ++j) { // Accumulate the error
                                    error += layers[l + 1].gradient[j] * weights[l](i, j);
                                }

                                layers[l].gradient[i] = error * layers[l].applyActivationDerivative(layers[l].z[i]); // Get activation for the preactivation neauron
                            }
                        }
                    }
                }
            }

        }
};

#endif