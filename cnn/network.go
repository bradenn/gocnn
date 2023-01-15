package cnn

import "gonum.org/v1/gonum/mat"

type Network struct {
	layers []Layer
}

func NewNetwork(layer ...Layer) Network {
	network := Network{
		layers: layer,
	}
	return network
}

// FeedForward runs through the CNN with a provided input
func (n *Network) FeedForward(input *mat.Dense) error {
	return nil
}

// BackPropagate computes errors with a provided training matrix, which is uses to adjust the weights
func (n *Network) BackPropagate() error {
	return nil
}

// Print prints the output layer results to stdout
func (n *Network) Print() error {
	return nil
}
