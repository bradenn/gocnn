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
