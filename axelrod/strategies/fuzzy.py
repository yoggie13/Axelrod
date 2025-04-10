from axelrod.action import Action, actions_to_str
from axelrod.player import Player
from axelrod.strategy_transformers import (
    FinalTransformer,
    TrackHistoryTransformer,
)
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import numpy as np
from collections import Counter
import pandas as pd

C, D = Action.C, Action.D

class Fuzzy(Player):

    # These are various properties for the strategy
    name = "Fuzzy"
    classifier = {
        "memory_depth": float("inf"),
        "stochastic": False,
        "long_run_time": False,
        "inspects_source": False,
        "manipulates_source": False,
        "manipulates_state": False,
    }

    cooperation = ctrl.Antecedent(np.arange(0, 100, 1), 'cooperation')
    adaptivity = ctrl.Antecedent(np.arange(0, 100, 1), 'adaptivity')
    forgivness = ctrl.Antecedent(np.arange(0, 100, 1), 'forgivness')
    stochastic = ctrl.Antecedent(np.arange(0, 100, 1), 'stochastic')

    cooperation.automf(names=["low", "medium", "high"])
    adaptivity.automf(names=["no", "yes"])
    forgivness.automf(names=["low", "medium", "high"])
    forgivness['low'] = fuzz.gaussmf(forgivness.universe, 0, 25)
    forgivness['medium'] = fuzz.trimf(forgivness.universe, [25, 50, 75])
    stochastic.automf(names=["no", "sometimes", "always"])

    resulting_strategy = ctrl.Consequent(np.arange(0, 100, 1),'resulting_strategy')
    resulting_strategy['D'] = fuzz.trimf(resulting_strategy.universe, [0, 25, 50])
    resulting_strategy['C'] = fuzz.trimf(resulting_strategy.universe, [35, 75, 100])

    rule1 = ctrl.Rule(cooperation['high'] & adaptivity['no'] & (forgivness['medium'] | forgivness['high']), resulting_strategy['D'])
    rule2 = ctrl.Rule(forgivness['low'] & cooperation['high'], resulting_strategy['C'])
    rule3 = ctrl.Rule(stochastic['always'] | adaptivity['no'], resulting_strategy['D'])
    rule4 = ctrl.Rule(cooperation['low'] | (cooperation['medium'] & forgivness['low']), resulting_strategy['D'])
    rule5 = ctrl.Rule(cooperation['medium'] & forgivness['medium'] & adaptivity['yes'], resulting_strategy['C'])

    strategy_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5])
    chosen_strategy = ctrl.ControlSystemSimulation(strategy_ctrl)

    h = {'Name':'', 'Fuzzy' : [], 'Opponent': []}

    forgivness_index = 0
    stochastic_index = 0
    df = pd.DataFrame(columns=["Coop", "Adap", "Forg", "Stoch", "D", "C"])

    def calcCooperation(self, opponent):
        return (Counter(opponent.history)[C])/len(opponent.history)*100

    def calcAdaptivity(self, opponent):
        adapCounter = 0
        adapReaction = 0

        if(len(self.history) < 3): 
            return 0
        
        for i in range(3, len(self.history)):
            if (self.history[i-3] == C and self.history[i-2] == D):
                adapCounter += 1
                if (opponent.history[i-1] == D):
                    adapReaction += 1
                elif (self.history[i-1] == D and opponent.history[i] == D):
                    adapReaction += 0.5
            elif (self.history[i-3] == D and self.history[i-2] == C):
                adapCounter += 1
                if (opponent.history[i-1] == C):
                    adapReaction += 1
                elif (self.history[i-1] == C and opponent.history[i] == C):
                    adapReaction += 0.5

            # if (self.history[i-2] == C and self.history[i-1] == D):
            #     adapCounter += 1
            #     if (opponent.history[i] == D):
            #         adapReaction += 1
            # elif (self.history[i-2] == D and self.history[i-1] == C):
            #     adapCounter += 1
            #     if (opponent.history[i] == C):
            #         adapReaction += 1
            # elif (self.history[i-2] == C and self.history[i-1] == C):
            #     adapCounter += 1
            #     if (opponent.history[i] == D):
            #         adapReaction += 1
            # elif (self.history[i-2] == D and self.history[i-1] == D):
            #     adapCounter += 1
            #     if (opponent.history[i] == D):
            #         adapReaction += 1  
        if adapCounter == 0:
            return 0
        
        return adapReaction/adapCounter*100                     
    
    def calcForgivness(self, opponent):
        DCounter = 0
        punishmentCounter = 0

        for i in range(0, len(self.history)-1):
            if(self.history[i] == D):
                DCounter += 1
                for j in range (i+1, len(opponent.history)):
                    if(opponent.history[j] == C):
                        break
                    else:
                        punishmentCounter += 1

        if punishmentCounter > 0:
            return DCounter/punishmentCounter*100
        else:
            return 100
    
    def calcStochastic(self, opponent):
        patterns = [
            [C, C, C],
            [C, C, D],
            [C, D, C],
            [C, D, D],
            [D, C, C],
            [D, C, D],
            [D, D, C],
            [D, D, D]
        ]

        nonStochasticCounter = 0
        patternPlayedCounter = 0

        for p in patterns:
            opponentsReactions = []
            for i in range(0, len(self.history)-3):
                if ([self.history[i], self.history[i+1], self.history[i+2]] == p):
                    opponentsReactions.append([opponent.history[i+1], opponent.history[i+2], opponent.history[i+3]])
            
            unique_patterns = len(set(tuple(sub) for sub in opponentsReactions))

            if(len(opponentsReactions) > 0):
                nonStochasticCounter += 0 if unique_patterns == 1 else unique_patterns
                patternPlayedCounter += len(opponentsReactions)
        
        if patternPlayedCounter == 0:
            return 0
        
        return nonStochasticCounter/patternPlayedCounter*100

    def strategy(self, opponent: Player) -> Action:

        if(len(self.history) == 0 or D not in opponent.history):
            return C
        
        else:
            self.chosen_strategy.input['cooperation'] = self.calcCooperation(opponent)
            self.chosen_strategy.input['adaptivity'] = self.calcAdaptivity(opponent)
            self.chosen_strategy.input['forgivness'] = self.calcForgivness(opponent)
            self.chosen_strategy.input['stochastic'] = self.calcStochastic(opponent)
            
            self.chosen_strategy.compute()

            d_membership = fuzz.interp_membership(self.resulting_strategy.universe, self.resulting_strategy['D'].mf, self.chosen_strategy.output['resulting_strategy'])
            c_membership = fuzz.interp_membership(self.resulting_strategy.universe, self.resulting_strategy['C'].mf, self.chosen_strategy.output['resulting_strategy'])

            od = self.chosen_strategy._get_inputs()

            # self.df = pd.concat([self.df, pd.DataFrame({
            #         "Coop": [od['cooperation']],
            #         "Adap": [od['adaptivity']],
            #         "Forg": [od['forgivness']],
            #         "Stoch": [od['stochastic']],
            #         "D": [d_membership],
            #         "C": [c_membership],
            #         "Chosen": ["D" if d_membership >= 0.4 and c_membership < 0.6 else "C"]
            #     })], ignore_index=True)

            # if(len(self.history) == 199):
            #     print(self.df)

            if(d_membership >= 0.4 and c_membership < 0.6):
                return D
        
        return C
