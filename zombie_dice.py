from typing import List
import random as rn
from copy import deepcopy

WINNING_SCORE = 13
DICE_TO_DEAL = 3
SHOTS_UNTIL_DEAD = 3

class Dice:
    def __init__(self, sides: List[str], name: str):
        self.sides = sides
        self.name = name 

    def roll(self) -> str:
        return rn.choice(self.sides)
        

class Deck:
    def __init__(self, dices: List[Dice]):
        self.dices = dices

    def shuffle(self):
        rn.shuffle(self.dices)

    def reverse(self):
        self.dices.reverse()

    def deal_dice(self, num_dice: int) -> List[Dice]:
        if num_dice <= len(self.dices):
            dealt_dices = self.dices[len(self.dices) - num_dice:]
            # self.dices = self.dices[:len(self.dices)]
            for _ in range(num_dice):
                self.dices.pop() # We need to do it this way to modify list in place
            return dealt_dices
        else:
            raise Exception(f"Not enough in deck Error, in deck: {len(self.dices)}, requested: {num_dice}")

    def add_dice(self, dice: Dice):
        self.dices.append(dice)

    def add_deck(self, new_deck):
        self.dices.extend(new_deck.dices)

    def prepend(self, new_deck):
        # logic from golang implementation
        # self.dices = new_deck.dices + self.dices
        # we want to do a prepend in place
        self.reverse()
        new_deck.reverse()
        self.dices.extend(new_deck.dices)
        self.reverse()


class Player:
    def __init__(self, player_state, id, is_ai, total_score):
        self.player_state = player_state
        self.id = id
        self.is_ai = is_ai
        self.total_score = total_score

    def take_turn(self, deck):
        # deck will be modified in place

        turn_result = []

        if len(deck.dices) < 3:
            new_deck = init_zombie_deck()
            new_deck.shuffle()
            deck.prepend(new_deck)

        dices_to_roll = deck.deal_dice(3)

        self.player_state.walks_taken_last_roll = 0

        for roll_ind, d in enumerate(dices_to_roll):
            side = d.roll()

            turn_result.append((d.name, side))

            if side == "brain":
                self.player_state.current_score += 1
                self.player_state.brains_rolled += 1
            elif side == "shot":
                self.player_state.times_shot += 1
            elif side == "walk":
                deck.add_dice(d)
                self.player_state.walks_taken_last_roll += 1

        if self.player_state.times_shot >= 3:
            self.player_state.is_dead = True

        self.player_state.turns_taken += 1

        # print(f"result for player: {self.id} : {turn_result}")

        return turn_result
            
class PlayerState:
    def __init__(self, turns_taken, current_score, times_shot, brains_rolled, 
                       walks_taken_last_roll, is_dead):
                    self.turns_taken = turns_taken
                    self.current_score = current_score
                    self.times_shot = times_shot
                    self.brains_rolled = brains_rolled
                    self.walks_taken_last_roll = walks_taken_last_roll
                    self.is_dead = is_dead

    def reset(self):
        self.turns_taken = 0
        self.current_score = 0
        self.times_shot = 0
        self.walks_taken_last_roll = 0
        self.brains_rolled = 0
        self.is_dead = False

class PlayerTurnResult:
    def __init__(self, turn_result, round_score, times_shot, total_score, is_dead, winner, player_id, continue_turn):
        self.turn_result = turn_result
        self.round_score = round_score
        self.times_shot = times_shot
        self.total_score = total_score
        self.is_dead = is_dead
        self.winner = winner
        self.player_id = player_id
        self.continue_turn = continue_turn

                

class GameState:
    def __init__(self,
                    game_state_id,
                    players,
                    zombie_deck,
                    player_turn,
                    winner,
                    game_over,
                    is_active,
                    move_log
                    ):
        self.game_state_id = game_state_id
        self.players = players
        self.zombie_deck = zombie_deck
        self.player_turn = player_turn
        self.winner = winner
        self.game_over = game_over
        self.is_active = is_active
        self.move_log = move_log


    def end_turn(self):

        if not self.players[self.player_turn].player_state.is_dead: 
            self.players[self.player_turn].total_score += self.players[self.player_turn].player_state.current_score

        next_player_turn = self.player_turn + 1

        if next_player_turn >= len(self.players):
            next_player_turn = 0
            self.end_round()

        self.player_turn = next_player_turn

        deck = init_zombie_deck()
        deck.shuffle()

        self.zombie_deck = deck

    def end_round(self):

        player_score_to_count = {}
        max_score = 0
        for p in self.players:
            if p.total_score not in player_score_to_count:
                player_score_to_count[p.total_score] = 1
            else:
                player_score_to_count[p.total_score] += 1

            if p.total_score > max_score:
                max_score = p.total_score        

        if max_score >= 13 and player_score_to_count[max_score] == 1:
            for p in self.players:
                if p.total_score == max_score:
                    self.winner = p  
                    self.game_over = True


def init_zombie_deck():
    green = Dice(["shot", "walk", "walk", "brain", "brain", "brain"], "green")
    yellow = Dice(["shot", "shot", "walk", "walk", "brain", "brain"], "yellow")
    red = Dice(["shot", "shot", "shot", "walk", "walk", "walk"], "red")

    zombie_deck = Deck([])

    for i in range(6):
        zombie_deck.add_dice(deepcopy(green))

    for i in range(4):
        zombie_deck.add_dice(deepcopy(yellow))

    for i in range(3):
        zombie_deck.add_dice(deepcopy(red))

    return zombie_deck

def init_player_state():
    return PlayerState(0, 0, 0, 0, 0, False)

def init_game_state(players, game_state_id):

    deck = init_zombie_deck()
    deck.shuffle()

    return GameState(game_state_id, players, deck, 0, None, False, False, None)



if __name__ == "__main__":

    player_a = Player(init_player_state(), "a", False, 0)
    player_b = Player(init_player_state(), "b", False, 0)

    players = [player_a, player_b]

    game_state = init_game_state(players, "my_game")

    player_a.take_turn(game_state.zombie_deck)

    print(player_a.player_state.__dict__)
    print(f"n_dices: {len(game_state.zombie_deck.dices)}")

    import ipdb 
    ipdb.set_trace()