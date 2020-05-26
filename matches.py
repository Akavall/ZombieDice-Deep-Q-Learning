
from collections import Counter
import random as rn
from keras.models import load_model
import numpy as np

import torch
from pytorch_agent import RNNRegressor

from zombie_dice import Player, init_game_state, init_player_state
from learning import get_features



class RandomAI:

    def should_continue(self, player=None, game_state=None):
        return rn.choice([False, True])


class Greedy:

    def should_continue(self, player=None, game_state=None):
        if player.player_state.times_shot >= 2:
            return False
        else:
            return True

class RL_Simple:

    def __init__(self, model_path):
        self.model = load_model(model_path)

    def should_continue(self, player=None, game_state=None):

        state, _, _ = get_features(player, game_state)

        prediction = self.model.predict(np.expand_dims(state, axis=0))

        return np.argmax(prediction[0])

class RL_PyTorch:

    def __init__(self, model_path):
        self.model = RNNRegressor(7, 2)
        self.model.load_state_dict(torch.load("pytorch_model.pt"))
        self.model.eval()


    def should_continue(self, player=None, game_state=None):

        state, _, _ = get_features(player, game_state)

        state_torch = torch.from_numpy(np.expand_dims(state, axis=0)).float()
        prediction = self.model(state_torch)
        return prediction.argmax()




def one_match(ai_a, ai_b):

    player_a = Player(init_player_state(), "a", False, 0)
    player_b = Player(init_player_state(), "b", False, 0)
    game_state = init_game_state([player_a, player_b], "my_game")

    while game_state.game_over is False:

        while(ai_a.should_continue(player_a, game_state) and not player_a.player_state.is_dead):
            player_a.take_turn(game_state.zombie_deck)

        game_state.end_turn()
        player_a.player_state.reset()

        while(ai_b.should_continue(player_b, game_state) and not player_b.player_state.is_dead):
            player_b.take_turn(game_state.zombie_deck)

        game_state.end_turn()
        player_b.player_state.reset()

        game_state.end_round()

    return game_state.winner.id


if __name__ == "__main__":

    random_ai = RandomAI()
    greedy_ai = Greedy()
    # rl_simple_ai_a = RL_Simple("experimental_model.h5")
    # rl_simple_ai_b = RL_Simple("my_model_1000_4_layer_eps_06_4.h5")
    # rl_simple_ai_a = RL_Simple("experimental_model_method_0_model4.h5")

    
    # best model
    # rl_simple_ai_b = RL_Simple("new_best_model.h5")
    # rl_simple_ai_b = RL_Simple("exprimental_model_method_1_model_4.h5")

    # it looks like 1000 is slightly better than 5000

    pytorch_agent = RL_PyTorch("pytorch_model")
    keras_agent = RL_Simple("experimental_model.h5")

    n_matches = 1000

    match_result = [one_match(keras_agent, pytorch_agent) for _ in range(n_matches)]

    print(f"match stats: {Counter(match_result)}")





        






    
