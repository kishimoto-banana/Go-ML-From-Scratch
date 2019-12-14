package main

import (
	"fmt"
	"math/rand"

	"github.com/kishimoto-banana/Go-ML-From-Scratch/supervised"
	"gonum.org/v1/gonum/mat"
)

var (
	N    = 100
	nDim = 2
	k    = 5
)

func matPrint(X mat.Matrix) {
	fa := mat.Formatted(X, mat.Prefix(""), mat.Squeeze())
	fmt.Printf("%v\n", fa)
}

func h(x1, x2 float64) float64 {
	return 2*x1 - 3*x2 - 1
}

func main() {

	x1 := make([]float64, N)
	x2 := make([]float64, N)
	for i := 0; i < N; i++ {
		x1[i] = 10*rand.Float64() - 5
		x2[i] = 10*rand.Float64() - 5
	}
	X1 := mat.NewDense(N, 1, x1)
	X2 := mat.NewDense(N, 1, x2)
	X := mat.NewDense(N, 2, nil)
	X.Augment(X1, X2)

	t := make([]float64, N)
	for i := range t {
		if h(x1[i], x2[i]) >= 0 {
			t[i] = 1
		}
	}
	y := mat.NewDense(N, 1, t)

	knn := supervised.NewKNN(k)
	yPred := knn.Predict(X, X, y)
	// matPrint(y)
	matPrint(yPred)
}
