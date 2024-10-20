## Zombie Dice Reinforcement Learning 

### Approach 

In this implementation, we are only focusing on solving one term at a time, not the entire game. For example, the model does not take into account the total number of brains of the current player and the oppenent, only the brains and kills on the given turn. Of course, including this information would create a more powerful model, but for simplicity we took the simpler approach.

**Features:**

1. Current score (number of brains) on this turn 
2. Times shot this turn 
3. Number of green dices with Walk
4. Number of yellow dices with Walk
5. Number of red dices with Walk 
6. Number of greed dices left in the deck
7. Number of yellow dices left in the deck 
8. Number of red dices left in the deck 

We use these features to set up a Double Q-learning approach.

### How to Run

Step 1: Set environmental variable for the path where the model will be saved to. For example:

```
export PYTORCH_MODEL_PATH="pytorch_model.pt"
```

Step 2: Train the model:

```
python learning_pytorch.py
```

Step 3: Test the newly trained model against naive implementations:

```
python matches.py
```

The result might looks look something like this:

```
PyTorch AI vs Random: Counter({'a': 970, 'b': 30})
PyTorch AI vs Greedy (max 1 shot): Counter({'a': 733, 'b': 267})
PyTorch AI vs Greedy (max 2 shots): Counter({'a': 575, 'b': 425})
```

Which shows that our network can beat various naive approaches.