package supervised

import (
	"math"
	"sort"

	"gonum.org/v1/gonum/mat"
)

type argsort struct {
	s    []float64
	inds []int
}

func (a argsort) Len() int {
	return len(a.s)
}

func (a argsort) Less(i, j int) bool {
	return a.s[a.inds[i]] < a.s[a.inds[j]]
}

func (a argsort) Swap(i, j int) {
	a.inds[i], a.inds[j] = a.inds[j], a.inds[i]
}

func ArgsortNew(src []float64) []int {
	inds := make([]int, len(src))
	for i := range src {
		inds[i] = i
	}
	Argsort(src, inds)
	return inds
}

func Argsort(src []float64, inds []int) {
	if len(src) != len(inds) {
		panic("floats: length of inds does not match length of slice")
	}
	a := argsort{s: src, inds: inds}
	sort.Sort(a)
}

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
		sortedIndicies := ArgsortNew(distances)

		neighbors := []int{}
		for i := 0; i < knn.k; i++ {
			neighbors = append(neighbors, int(yTrain.At(sortedIndicies[i], 0)))
		}
		yPred.Set(i, 0, float64(knn.Vote(neighbors)))
	}
	return yPred
}

func (knn *KNN) Vote(neighbors []int) int {
	voteMap := make(map[int]int)
	for _, label := range neighbors {
		voteMap[label] = voteMap[label] + 1
	}

	var maxClass int
	maxCount := 1
	for label, count := range voteMap {
		if count > maxCount {
			maxCount = count
			maxClass = label
		}
	}
	return maxClass
}
