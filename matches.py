
from collections import Counter
import random as rn
from keras.models import load_model
import numpy as np

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



def one_match(ai_a, ai_b):

    player_a = Player(init_player_state(), "a", False, 0)
    player_b = Player(init_player_state(), "b", False, 0)
    game_state = init_game_state([player_a, player_b], "my_game")

    while game_state.game_over is False:

        while(ai_a.should_continue(player_a, game_state)):
            player_a.take_turn(game_state.zombie_deck)

        game_state.end_turn()
        player_a.player_state.reset()

        while(ai_b.should_continue(player_b, game_state)):
            player_b.take_turn(game_state.zombie_deck)

        game_state.end_turn()
        player_b.player_state.reset()

        game_state.end_round()

    return game_state.winner.id


if __name__ == "__main__":

    random_ai = RandomAI()
    greedy_ai = Greedy()
    rl_simple_ai_a = RL_Simple("experimental_model.h5")
    # rl_simple_ai_b = RL_Simple("my_model_1000_4_layer_eps_06_4.h5")

    
    # best model
    # rl_simple_ai_a = RL_Simple("my_model_1000_4_layer.h5")
    rl_simple_ai_b = RL_Simple("my_model_1000_4_layer.h5")

    # it looks like 1000 is slightly better than 5000

    n_matches = 1000

    match_result = [one_match(rl_simple_ai_a, rl_simple_ai_b) for _ in range(n_matches)]

    print(f"match stats: {Counter(match_result)}")





        






    
