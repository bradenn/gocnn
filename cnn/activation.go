package cnn

import "math"

type Activation interface {
	Activate(float64) float64
	ActivateDx(float64) float64
	Name() string
}

type Sigmoid struct{}

func (s *Sigmoid) Activate(input float64) float64 {
	return 1.0 / (1.0 + math.Exp(input*-1.0))
}

func (s *Sigmoid) ActivateDx(input float64) float64 {
	return 1 * (1 - input)
}

func (s *Sigmoid) Name() string {
	return "Sigmoid"
}

type Tanh struct{}

func (t *Tanh) Activate(input float64) float64 {
	return math.Tanh(input)
}

func (t *Tanh) ActivateDx(input float64) float64 {
	tanh := math.Tanh(input)
	return 1.0 - tanh*tanh
}

func (t *Tanh) Name() string {
	return "Tanh"
}

type ReLU struct{}

func (r *ReLU) Activate(input float64) float64 {
	if input > 0.0 {
		return input
	}
	return 0.0
}

func (r *ReLU) ActivateDx(input float64) float64 {
	if input > 0.01 {
		return 0.9999
	}
	return 0.0001
}

func (r *ReLU) Name() string {
	return "ReLU"
}

type LReLU struct{}

func (l *LReLU) Activate(input float64) float64 {
	if input > 0.0 {
		return input
	}
	return 0.01 * input
}

func (l *LReLU) ActivateDx(input float64) float64 {
	if input > 0.0 {
		return 1.0
	}
	return 0.01
}

func (l *LReLU) Name() string {
	return "LReLU"
}
