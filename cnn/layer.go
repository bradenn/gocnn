package cnn

import (
	"fmt"
	"gonum.org/v1/gonum/mat"
)

type Layer interface {
	// InputSize returns the x,y,z dimensions of the incoming layer
	InputSize() Dimensions
	// OutputSize returns the x,y,z dimensions of the outgoing layer
	OutputSize() Dimensions
	// ActivatedOutputs returns the layer's activated outputs for the next layer
	ActivatedOutputs() *mat.Dense
	// Weights returns the matrix of the layer's weights
	Weights() *mat.Dense
	// SetWeights changes the layer's weights to the provided ones
	SetWeights(*mat.Dense)
	// Gradients returns the layer's gradients matrix
	Gradients() *mat.Dense
	// FeedForward passes data through the layer
	FeedForward(inputs *mat.Dense) error
	// ComputeGradients computes the gradients based on the error generator
	ComputeGradients(errors *mat.Dense) error
	// UpdateWeights changes the weights based on the errors
	UpdateWeights(rate float64) error
	// PrintOutput prints the output matrix of the layer
	PrintOutput()
	// PrintWeights prints the weights matrix of the layer
	PrintWeights()
	// Stride returns the layer's stride value
	Stride() int
	// Type returns the layer's type (MaxPooling, Convolution, etc.)
	Type() string
	// SetActivation changes the layer's activation function and derivative function
	SetActivation(activation Activation)
}

type DefaultLayer struct {
	inputs  *mat.Dense
	outputs *mat.Dense
	biases  *mat.Dense

	weights     *mat.Dense
	deltas      *mat.Dense
	derivatives *mat.Dense

	name string

	activation Activation
}

func (d *DefaultLayer) InputSize() Dimensions {
	r, c := d.inputs.Dims()
	return Dimensions{X: r, Y: c, Z: 1}
}

func (d *DefaultLayer) OutputSize() Dimensions {
	r, c := d.outputs.Dims()
	return Dimensions{X: r, Y: c, Z: 1}
}

func (d *DefaultLayer) ActivatedOutputs() *mat.Dense {
	//TODO implement me
	panic("implement me")
}

func (d *DefaultLayer) Weights() *mat.Dense {
	return d.weights
}

func (d *DefaultLayer) Gradients() *mat.Dense {
	//TODO implement me
	panic("implement me")
}

func (d *DefaultLayer) FeedForward(inputs *mat.Dense) error {
	//TODO implement me
	panic("implement me")
}

func (d *DefaultLayer) ComputeGradients(errors *mat.Dense) error {
	//TODO implement me
	panic("implement me")
}

func (d *DefaultLayer) UpdateWeights(rate float64) error {
	//TODO implement me
	panic("implement me")
}

func (d *DefaultLayer) SetWeights(dense *mat.Dense) {
	d.weights = dense
}

func (d *DefaultLayer) PrintOutput() {
	rows, cols := d.outputs.Dims()
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			fmt.Printf("%.02f", d.outputs.At(i, j))
		}
		fmt.Printf("\n")
	}
}

func (d *DefaultLayer) PrintWeights() {
	rows, cols := d.weights.Dims()
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			fmt.Printf("%.02f", d.weights.At(i, j))
		}
		fmt.Printf("\n")
	}
}

func (d *DefaultLayer) Stride() int {
	return 0
}

func (d *DefaultLayer) Type() string {
	return "default"
}

func (d *DefaultLayer) SetActivation(activation Activation) {
	d.activation = activation
}

func NewLayer(name string, input Dimensions, weights []float64, biases []float64) DefaultLayer {
	layer := DefaultLayer{
		name:        name,
		inputs:      mat.NewDense(input.Z*input.X, input.Y, nil),
		outputs:     &mat.Dense{},
		deltas:      &mat.Dense{},
		derivatives: &mat.Dense{},
		activation:  &Sigmoid{},
	}
	//// Load in pretrained weights
	//layer.weights = mat.NewDense(inputs, outputs, weights)
	//// Initialize biases
	//layer.biases = mat.NewDense(1, outputs, nil)
	//// Populate the weight layer if no weights are provided
	//if weights == nil {
	//	populateMatUniform(layer.weights)
	//}
	//// Populate the biases layer if none are provided
	//if biases == nil {
	//	populateMatUniform(layer.biases)
	//}

	return layer
}
