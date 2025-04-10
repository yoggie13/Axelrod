from axelrod.action import Action, actions_to_str
from axelrod.player import Player
from axelrod.strategy_transformers import (
    FinalTransformer,
    TrackHistoryTransformer,
)
import random

C, D = Action.C, Action.D

class MAB(Player):

    # These are various properties for the strategy
    name = "MAB"
    classifier = {
        "memory_depth": float("inf"),
        "stochastic": True,
        "long_run_time": False,
        "inspects_source": False,
        "manipulates_source": False,
        "manipulates_state": False,
    }

    avgC = 0.00
    avgD = 0.00
    rand = 0

    h = {'Name':'', 'MAB' : [], 'Opponent': []}
    def calcAVG(self, opponent):
        sumC = 0
        iterC = 0
        sumD = 0
        iterD = 0

        for i in range(0, len(self.history)):
            if self.history[i] == C:
                iterC += 1
                if opponent.history[i] == C:
                    sumC += 3
            else:
                iterD += 1
                if opponent.history[i] == C:
                    sumD += 5
                else:
                    sumD += 1

        if iterC > 0:
            self.avgC = round(sumC / iterC,2)
        if iterD > 0:
            self.avgD = round(sumD / iterD,2)
                        
    def strategy(self, opponent: Player) -> Action:
        #probati lookahead mab algoritam sa paternima
        self.calcAVG(opponent)
        r = random.random()
        l = len(self.history)
        if not self.history:
            return C
        if l == 1:
            return D
        if l <= 20 and r < 0.2:
            if self.avgC >= self.avgD:
                return D
            if self.avgC < self.avgD:
                return C
        if l <= 50 and r < 0.1:
            if self.avgC >= self.avgD:
                return D
            if self.avgC < self.avgD:
                return C
        if l > 50 and r < 0.05:
            if self.avgC >= self.avgD:
                return D
            if self.avgC < self.avgD:
                return C
        if 1.25*self.avgC >= self.avgD:
            return C
        if 1.25*self.avgC < self.avgD:
            return D
        # React to the opponent's last move
       
        return C
