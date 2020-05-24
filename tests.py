
from keras.models import load_model
import torch
import numpy as np


def test_prediction(model, state, expected_move, pytorch=False):

    # state = [
    # current_score,
    # shot,
    # brains,
    # walks,
    # green,
    # yellow,
    # red
    #        ]

    if not pytorch:
        prediction = model.predict(np.expand_dims(state, axis=0))
        should_move = np.argmax(prediction[0])
    else:
        state_torch = torch.from_numpy(np.expand_dims(state, axis=0)).float() 
        prediction = model(state_torch)
        should_move = prediction.argmax()

    return expected_move == should_move



def test_model(model, pytorch=False, verbose=False):

    score = 0

    state = [0,0,0,0,6,4,3]

    result = test_prediction(model, state, True, pytorch)


    if verbose:
        if not result:
            print("missed test 1")
        else:
            print("passed test 1")

    score += result

    state = [4,2,4,0,0,4,3]

    result = test_prediction(model, state, False, pytorch)

    if verbose:
        if not result:
            print("missed test 2")
        else:
            print("passed test 2")

    score += result

    state = [7,2,4,0,0,1,3]

    result = test_prediction(model, state, False, pytorch)

    if verbose:
        if not result:
            print("missed test 3")
        else:
            print("passed test 3")

    score += result

    state = [2,0,2,1,6,4,1]

    result = test_prediction(model, state, True, pytorch)

    if verbose:
        if not result:
            print("missed test 4")
        else:
            print("passed test 4")

    score += result

    state = [4,0,4,2,6,2,1]

    result = test_prediction(model, state, True, pytorch)

    if verbose:
        if not result:
            print("missed test 5")
        else:
            print("passed test 5")

    score += result

    state = [6,0,6,3,6,1,0]

    result = test_prediction(model, state, True, pytorch)

    if verbose:
        if not result:
            print("missed test 6")
        else:
            print("passed test 6")

    score += result   
    
    state = [0,2,0,1,4,4,3]

    result = test_prediction(model, state, True, pytorch)

    if verbose:
        if not result:
            print("missed test 7")
        else:
            print("passed test 7")

    score += result

    state = [2,2,2,2,0,4,3]

    result = test_prediction(model, state, False, pytorch)

    if verbose:
        if not result:
            print("missed test 8")
        else:
            print("passed test 8")

    score += result

    state = [6,1,6,2,0,3,3]

    result = test_prediction(model, state, False, pytorch)

    if verbose:
        if not result:
            print("missed test 9")
        else:
            print("passed test 9")

    score += result

    state = [5,1,5,3,0,4,3]

    result = test_prediction(model, state, False, pytorch)

    if verbose:
        if not result:
            print("missed test 10")
        else:
            print("passed test 10")

    score += result

    state = [1,1,1,1,3,4,3]

    result = test_prediction(model, state, True, pytorch)

    if verbose:
        if not result:
            print("missed test 11")
        else:
            print("passed test 11")

    score += result

    state = [1,2,1,0,3,4,3]

    result = test_prediction(model, state, False, pytorch)

    if verbose:
        if not result:
            print("missed test 12")
        else:
            print("passed test 12")

    score += result


    state = [7,0,7,3,6,0,0]

    result = test_prediction(model, state, True, pytorch)

    if verbose:
        if not result:
            print("missed test 13")
        else:
            print("passed test 13")

    score += result  

    state = [6,0,6,3,4,2,0]

    result = test_prediction(model, state, True, pytorch)

    if verbose:
        if not result:
            print("missed test 14")
        else:
            print("passed test 14")

    score += result  





    return score


if __name__ == "__main__":

    # model = load_model("new_best_model.h5")
    # model = load_model("my_model_100.h5")
    # model = load_model("experimental_model_method_0_model4.h5")
    model = load_model("experimental_model.h5")

    score = test_model(model, verbose=True)

    print(f"total_score: {score}")
    





