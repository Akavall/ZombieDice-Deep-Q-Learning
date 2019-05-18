package main

import (
	"fmt"

	tensorflow "github.com/tensorflow/tensorflow/tensorflow/go"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
)

func makeTensor(data [][]float32) (*tf.Tensor, error) {
	return tf.NewTensor(data)
}

func getResult(prediction [][]float32) bool {
	if prediction[0][0] > prediction[0][1] {
		return false
	}
	return true
}

type RL_AI struct {
	model *tensorflow.SavedModel
}

func (rl *RL_AI) load_model() {
	model, err := tf.LoadSavedModel("golang_model", []string{"tags"}, nil)

	rl.model = model

	if err != nil {
		fmt.Printf("Error loading saved model: %s\n", err.Error())
		return
	}

}

func (rl *RL_AI) predict(data [][]float32) bool {

	// defer rl.model.Session.Close()

	tensor, err := makeTensor(data)

	fmt.Println("Made Tensor")

	if err != nil {
		fmt.Println(err)
	}

	result, runErr := rl.model.Session.Run(
		map[tf.Output]*tf.Tensor{
			rl.model.Graph.Operation("dense_1_input").Output(0): tensor,
		},
		[]tf.Output{
			rl.model.Graph.Operation("my_output/BiasAdd").Output(0),
		},
		nil,
	)

	if runErr != nil {
		fmt.Println("ERROR!!! ", runErr)
	}

	fmt.Println("Result: ", result[0].Value())

	temp := result[0].Value().([][]float32)

	return getResult(temp)

	// fmt.Println(temp)
	// return true

}

func main() {

	rl_ai := RL_AI{}
	rl_ai.load_model()

	// defer rl.model.Session.Close()

	data := [][]float32{{1, 0, 0, 0, 5, 4, 3}}
	prediction := rl_ai.predict(data)

	fmt.Println("prediction ", prediction)

	data_2 := [][]float32{{4, 2, 4, 0, 0, 4, 3}}
	prediction_2 := rl_ai.predict(data_2)
	fmt.Println("prediction 2 ", prediction_2)

}
