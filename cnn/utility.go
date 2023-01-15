package cnn

import (
	"gonum.org/v1/gonum/mat"
	"math/rand"
)

func populateMatUniform(dense *mat.Dense) {
	rows, cols := dense.Dims()
	norm := float64(rows * cols)
	for i := 0; i < rows; i++ {
		for j := 0; j < rows; j++ {
			dense.Set(i, j, rand.NormFloat64()/norm)
		}
	}
}

// flatten takes in a matrix and returns a matrix with one row containing all data
func flatten(input *mat.Dense) *mat.Dense {
	// Get the dimensions of the input matrix
	rows, cols := input.Dims()
	// Compute the number of elements within the matrix
	quanta := rows * cols
	// Initialize an array to hold the values from the input
	data := make([]float64, quanta)
	// Pull the data from the input matrix and put it in the new array
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			data[i*cols+j] = input.At(i, j)
		}
	}
	// return a new matrix with the flattened data
	return mat.NewDense(1, quanta, data)
}
