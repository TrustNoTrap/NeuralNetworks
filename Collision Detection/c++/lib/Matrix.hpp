#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <vector>
#include <stdexcept>
#include <random>

class Matrix {
    private:
        std::vector<std::vector<double>> data; // A matrix of doubles; [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0] ...etc]
        size_t rows, cols;

    public:
        Matrix(size_t r, size_t c) : rows(r), cols(c) {
            data.resize(r, std::vector<double>(c, 0.0)); // Ensures the matrix that is being created will be initialized with 0 values
        }

        // Get size of rows / columns
        size_t getRows() const { return rows; }
        size_t getCols() const { return cols; }

        double& operator()(size_t i, size_t j) { // Read / write access
            if (i >= getRows() || j >= getCols()) {
                throw std::out_of_range("Index out of bounds");
            }

            return data[i][j];
        }

        // First const -> indicates that the return value cannot be modified
        // Second const -> indicates that the applies to the object itself on which the member function is called.
        // It signifies that the function will not modify any of the non-static data members of the class instance.
        // This is crucial for "const-correctness" as it allows the function to be called on const objects.
        const double& operator()(size_t i, size_t j) const { // Read only access
            if (i >= getRows() || j >= getCols()) {
                throw std::out_of_range("Index out of bounds");
            }

            return data[i][j];
        }

        Matrix& fillRandom(double min = -1.0, double max = 1.0) { // Fill matrix with random values
            std::random_device rd; // Random device
            std::mt19937 gen(rd()); // Number generator
            std::uniform_real_distribution<> dist(min, max);

            for (size_t i = 0; i < (*this).getRows(); ++i) {
                for (size_t j = 0; j < (*this).getCols(); ++j) {
                    (*this)(i, j) = dist(gen);
                }
            }

            return *this; // Return ref to this object
        }

        Matrix operator*(double scalar) const { // Create a new matrix multiplied by the scalar -> Does not change the original matrix
            Matrix result(getRows(), getCols());

            for (size_t i = 0; i < getRows(); ++i) {
                for (size_t j = 0; j < getCols(); ++j) {
                    result(i, j) = (*this)(i,j) * scalar;
                }
            }

            return result;
        }

        // Scale matrix in place
        Matrix& operator*=(double scalar) { // Instead of returning a new matrix, we modify the existing one (in the object)
            for (size_t i = 0; i < getRows(); ++i) {
                for (size_t j; j < getCols(); ++j) {
                    (*this)(i, j) *= scalar;
                }
            }

            return *this;
        }
};

#endif