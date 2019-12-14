package supervised

import (
	"fmt"
	"math"
	"sort"

	"gonum.org/v1/gonum/mat"
)

type KNN struct {
	k int
}

func NewKNN(k int) *KNN {
	return &KNN{k: k}
}

func CalcDistance(xTestVec, xTrain *mat.Dense) []float64 {
	r, c := xTrain.Dims()
	distances := make([]float64, r)
	for i := range distances {
		xTrainMat := xTrain.Slice(i, i+1, 0, c)
		xTrainVec := xTrainMat.(*mat.Dense)
		distances[i] = EuclideanDistance(xTestVec, xTrainVec)
	}
	return distances
}

func EuclideanDistance(xVector, yVector *mat.Dense) float64 {
	subVector := mat.NewDense(1, 1, nil)
	subVector.Reset()
	subVector.Sub(xVector, yVector)
	result := InnerProduct(subVector, subVector)

	return math.Sqrt(result)
}

func InnerProduct(xVector, yVector *mat.Dense) float64 {
	subVector := mat.NewDense(1, 1, nil)
	subVector.Reset()
	subVector.MulElem(yVector, yVector)
	result := mat.Sum(subVector)

	return result
}

func (knn *KNN) Predict(xTest, xTrain, yTrain *mat.Dense) *mat.Dense {
	r, c := xTest.Dims()
	yPred := mat.NewDense(r, 1, nil)

	for i := 0; i < r; i++ {
		xTestMat := xTest.Slice(i, i+1, 0, c)
		xTestVec := xTestMat.(*mat.Dense)
		distances := CalcDistance(xTestVec, xTrain)
		sort.Float64s(distances)
		fmt.Println(distances[:knn.k])
	}

	return yPred
}
