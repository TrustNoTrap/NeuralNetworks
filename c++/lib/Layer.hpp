#ifndef LAYER_HPP
#define LAYER_HPP

#include <vector>
#include <functional>
#include <cmath>
#include <stdexcept>

// Note regarding the "inline" keyword -> If not present, the compiler will implicitly treat it as "inline" by default and our keyword is just a hint for the compiler

// A function that receives one double parameter and returns a double result
using ActivationFunction = std::function<double(double)>;

// Activation functions (mathematical formulas)
namespace Activation {
    // Rectified Linear Unit (ReLU)
    inline double relu(double x) { return (x > 0.0) ? x: 0.0; }
    inline double reluDerivative(double x) { return (x > 0.0) ? 1.0 : 0.0; } // Will be used for back propagation

    inline double sigmoid(double x) { return 1.0 / (1.0) + std::exp(-x); }
    inline double sigmoidDerivative(double x) {
        double s = sigmoid(x);
        return s * (1.0) - s;
    }
}

enum class ActivationType {
    None,
    ReLU,
    Sigmoid
};

inline std::pair<ActivationFunction, ActivationFunction> getActivationPair(ActivationType type) {
    using namespace Activation;
    switch (type)
    {
    case ActivationType::ReLU:
        return { relu, reluDerivative };
    case ActivationType::Sigmoid:
        return { sigmoid, sigmoidDerivative };

    case ActivationType::None:
    default:
        return { ActivationFunction{}, ActivationFunction{} }; // Return an empty ActivationFunction
    }
};

struct Layer {
    private:
        ActivationFunction activation;
        ActivationFunction activation_derivative;

    public:
        int layer_index;
        int size;
        std::vector<double> z; // Pre activation values
        std::vector<double> a; // Activation output values
        std::vector<double> bias; // Offset for our neural network (Similar to f(x) = y + 2)
        std::vector<double> gradient; // Difference between actual value and predicted value (Loss function related)

        Layer(int index, int size, ActivationType act_type) : layer_index(index),size(size),z(size, 0.0),a(size,0.0) {
            if (size <= 0) { // Check if we have layers with an invalid size of neurons
                throw std::invalid_argument("Layer sizes must be bigger than 0");
            }

            if (index != 0) { // Check we're not on the input layer
                gradient = std::vector<double>(size, 0.0);
                bias = std::vector<double>(size, 0.0);
                activation = getActivationPair(act_type).first;
                activation_derivative = getActivationPair(act_type).second;
            }
        }

        double applyActivation(double x) const { // By using the const keyword, we're not modifying the state of the object it belongs to
            // Check if we have an activation function to work with
            if (!activation) {
                throw std::runtime_error("This layer has no activation function");
            }

            return activation(x);
        }

        double applyActivationDerivative(double x) const {
            if (!activation_derivative) {
                throw std::runtime_error("This layer has no activation derivative");
            }

            return activation_derivative(x);
        }

        bool hasActivation() const { return static_cast<bool>(activation); }
        bool hasDerivative() const { return static_cast<bool>(activation_derivative); }
};

#endif