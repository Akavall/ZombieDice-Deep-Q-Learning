
import numpy as np
import torch

from zombie_dice import Player, init_player_state, init_game_state
from pytorch_agent import PyTorchAgent

from collections import deque

from tests import test_model


def get_features(player, game_state):
    s = player.player_state 
    state = []
    state.append(s.current_score)
    state.append(s.times_shot)
    state.append(s.brains_rolled)
    state.append(s.walks_taken_last_roll)
    # state.append(s.is_dead)

    dices = game_state.zombie_deck.dices 

    greens = sum(x.name == "green" for x in dices)
    yellows = sum(x.name == "yellow" for x in dices)
    reds = sum(x.name == "red" for x in dices)

    if len(dices) >= 3:
        state.append(greens)
        state.append(yellows)
        state.append(reds)
    else:
        state.append(greens + 6)
        state.append(yellows + 4)
        state.append(reds + 3)

    if s.is_dead:
        reward = 0
    else:
        reward = s.brains_rolled

    return state, reward, s.is_dead


def learn(agent, n_episodes, maxlen_scores):

    scores = deque(maxlen=maxlen_scores)

    for e in range(n_episodes):

        print(f"On episode: {e}")

        player_a = Player(init_player_state(), "a", False, 0)
        game_state = init_game_state([player_a], "my_game")
        state, reward, done = get_features(player_a, game_state)
        state = np.expand_dims(state, axis=0)
        done = False

        while done is False:

            action = agent.act(state)
            if action == 0:
                done = True
                new_state = player_a.player_state.brains_rolled
                reward = new_state
            else:
                player_a.take_turn(game_state.zombie_deck)
                new_state, reward, done = get_features(player_a, game_state)
                reward = 0
                if done:
                    new_state = reward # reward is 0 here
                else:
                    new_state = np.expand_dims(new_state, axis=0)

            agent.remember((state, action, reward, new_state, done))

            state = new_state

            if done:
                agent.update_policy_weights()
                break

            agent.replay_memory(e)

        scores.append(new_state)
        print(f"score last 100 avg: {np.mean(scores)}, current_score: {new_state}")

        model_score = test_model(agent.model, pytorch=True)
        print(f"model_score: {model_score}")


if __name__ == "__main__":

    pytorch_agent = PyTorchAgent(input_size=7, action_size=2)

    learn(agent=pytorch_agent, n_episodes=1000, maxlen_scores=100)

    torch.save(pytorch_agent.model.state_dict(), "pytorch_model.pt")
