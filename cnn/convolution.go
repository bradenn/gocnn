package cnn

import (
	"gonum.org/v1/gonum/mat"
	"math/rand"
)

type Convolution struct {
	DefaultLayer

	// filters are the individual matrices used to identify features within the inputs
	filters []*mat.Dense
	// previous refers to the previous delta of the filter
	previous []*mat.Dense

	// stride represents the step size of the convolution
	stride int
	// filterSize represents the size of the kernels
	filterSize int

	// inputDims represent the size of the incoming layer (X, Y) * Z, where Z is channels
	inputDims Dimensions
	// outputDims represent the size of the outgoing layer
	outputDims Dimensions
}

func NewConvolutionLayer(input Dimensions, stride, filterSize, numFilters int) Layer {
	// Calculate the size of the output layer width using ((X-k)/S + 1)
	outputX := (input.X-filterSize)/stride + 1
	// Calculate the size of the output layer height using ((Y-k)/S + 1)
	outputY := (input.Y-filterSize)/stride + 1
	// Create the Convolution layer object and initialize the base layer
	layer := Convolution{
		// Initialize the base layer to handle the default functions
		DefaultLayer: DefaultLayer{
			inputs:  mat.NewDense(input.Z*input.X, input.Y, nil),
			outputs: mat.NewDense(outputX*numFilters, outputY, nil),
		},
		// Define the stride and filter size
		stride:     stride,
		filterSize: filterSize,
		// Initialize filter arrays
		filters:  make([]*mat.Dense, numFilters),
		previous: make([]*mat.Dense, numFilters),
		// Define the input and output dimensions
		inputDims:  input,
		outputDims: Dimensions{X: outputX, Y: outputY, Z: numFilters},
	}
	// Initialize each filter
	for i := 0; i < numFilters; i++ {
		switch input.Z {
		// In the case that the number of input channels is one, initialize normally
		case 1:
			// Initialize the filter matrix for the filter at index i
			layer.filters[i] = mat.NewDense(filterSize, filterSize, nil)
			// Populate the matrix with random uniform values
			for j := 0; j < filterSize; j++ {
				for k := 0; k < filterSize; k++ {
					// Set the given space to a random number between -0.5 and 0.5
					layer.filters[i].Set(j, k, rand.Float64()-0.5)
				}
			}
			// Initialize the previous delta layer
			layer.previous[i] = mat.NewDense(filterSize, filterSize, nil)
			// Populate the previous delta layer to all zeros
			layer.previous[i].Zero()
			// Continue to the next iteration of the loop
			continue
		// If the number of channels is greater than one, the configuration will be more advanced
		default:
			// Initialize the filter matrix to store each filter in one row, and each filter in a different column
			layer.filters[i] = mat.NewDense(filterSize*filterSize, input.Z, nil)
			// Populate this matrix in a similar fashion to the case of 1, but with the adjusted matrix
			for j := 0; j < filterSize*filterSize; j++ {
				for k := 0; k < input.Z; k++ {
					// Set the given space to a random number between -0.5 and 0.5
					layer.filters[i].Set(j, k, rand.Float64()-0.5)
				}
			}
			// Initialize the previous delta layer
			layer.previous[i] = mat.NewDense(filterSize*filterSize, input.Z, nil)
			// Populate the previous delta layer to all zeros
			layer.previous[i].Zero()
		}
	}
	// return the new layer
	return &layer
}

// OutputSize overrides the base output to account for the Convolution structure
func (c *Convolution) OutputSize() Dimensions {
	return c.outputDims
}

// Stride overrides the base function to provide the Convolution stride
func (c *Convolution) Stride() int {
	return c.stride
}

// Type overrides the base function to provide the Convolution string
func (c *Convolution) Type() string {
	return "convolution"
}
