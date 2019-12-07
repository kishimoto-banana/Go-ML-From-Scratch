package supervised

import (
	"math"
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

func Sigmoid(x *mat.Dense) *mat.Dense {
	sigmoid := func(i, j int, v float64) float64 {
		return 1 / (1 + math.Exp(-v))
	}
	r, c := x.Dims()
	result := mat.NewDense(r, c, nil)
	result.Apply(sigmoid, x)
	return result
}

// TODO: Add regularization term
type LogisticRegression struct {
	w            *mat.Dense
	learningRate float64
	nIterations  int
}

func NewLogisticRegression(n_dim int, learningRate float64, nIterations int) *LogisticRegression {
	wInit := make([]float64, n_dim)
	for i := range wInit {
		wInit[i] = rand.Float64()
	}

	return &LogisticRegression{
		w:            mat.NewDense(n_dim, 1, wInit),
		learningRate: learningRate,
		nIterations:  nIterations,
	}
}

// TODO: Convergence judgment
func (lr *LogisticRegression) Fit(X *mat.Dense, y *mat.Dense) {
	for i := 0; i < lr.nIterations; i++ {
		r, c := X.Dims()
		t := mat.NewDense(r, 1, nil)
		t.Product(X, lr.w)
		yPred := Sigmoid(t)
		r, c = yPred.Dims()
		yDiff := mat.NewDense(r, c, nil)
		yDiff.Sub(yPred, y)
		r, c = lr.w.Dims()
		grad := mat.NewDense(r, c, nil)
		grad.Product(X.T(), yDiff)
		lr.w.Sub(lr.w, grad)
	}
}

func (lr *LogisticRegression) Predict(X *mat.Dense) *mat.Dense {
	r, _ := X.Dims()
	t := mat.NewDense(r, 1, nil)
	t.Product(X, lr.w)
	yPred := Sigmoid(t)

	return yPred
}
