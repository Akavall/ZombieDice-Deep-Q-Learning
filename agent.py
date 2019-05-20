from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras import backend as K
from keras import regularizers 
import tensorflow as tf

from collections import deque
import random as rn
import random
import numpy as np


class Agent(object):

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = 0.001
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.memories_capacity = 1000
        self.memories = deque([], self.memories_capacity)
        self.batch_size = 32
        self.exploration_decay = 0.98
        self.discount_rate = 0.95
        self.epsilon = 0.5
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.update_policy_weights()

    def build_model(self):

        model = Sequential()
        model.add( Dense(24, input_dim=self.state_size, activation="relu"))

        # kernel_regularizer=regularizers.l2(0.05) if we ever need a regularization
        model.add( Dense(24, activation="relu"))
        model.add( Dense(24, activation="relu"))
        # model.add( Dense(24, activation="relu"))
        # model.add( Dense(24, activation="relu"))
        model.add( Dense(self.action_size, activation="linear", name="my_output"))

        model.compile(loss="mse", optimizer=Adam(lr=self.learning_rate))
        
        return model

    def remember(self, memory):
        
        self.memories.append(memory) 
   
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return rn.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0]) # returns action 


    def replay_memory(self):

        if len(self.memories) < self.batch_size:
            return

        sample_memories = rn.sample(self.memories, self.batch_size)

        for state, action, reward, new_state, done in sample_memories:
            
            # Why is this just state? Not state and Action?
            # At this point we get entire action_space
            # and we update action part later
            # https://keon.io/deep-q-learning/

            action_scores = self.model.predict(state)

            if done:
                action_scores[0][action] = reward
            else:

                # What would happen if we used self.model here?
                # The problem is explained here: http://deeplizard.com/learn/video/xVkPh9E9GfE
                # and here https://ai.stackexchange.com/questions/6982/why-does-dqn-require-two-different-networks
                # I think (mostly based on SE Question)
                # If let's say we get a very bad estimate of .predict(new_state)
                # Then this will be reflected in .predict(state)
                # Since this is the same model now, estimates for .predict(state) and .predict(new_state)
                # if .predict(new_state) is done by a different model, then at least this model has not
                # been trained on biased results, and things are more stable
                if isinstance(new_state, int):
                    # this happens player chose to end the turn
                    new_action_score = new_state
                else:
                    temp = self.target_model.predict(new_state)
                    # temp = self.model.predict(new_state) # experimental line
                    # I tested this, and it is much, much worse. Basically does not work.
                    new_action_score = np.max(temp)

                # This is just ignoring learning rate; learning rate is 1
                # target_q[0][action] = (reward + self.discount_rate * new_q)

                action_scores[0][action] = reward + self.discount_rate * new_action_score

            # Since the model given the state, gives back action scores
            # We train it the same way, we give it state, and 
            # modified actions scores to what it returned                 

            self.model.fit(state, action_scores, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_policy_weights(self):
        
        self.target_model.set_weights(self.model.get_weights())