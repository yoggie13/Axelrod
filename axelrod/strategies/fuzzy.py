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
from math import exp
from openpyxl import load_workbook
import axelrod.WriteToExcel as WriteToExcel
import torch
import joblib
import numpy as np


C, D = Action.C, Action.D

class FuzzyMethods():
    @staticmethod
    def calcCooperation(self, opponent):
        return (Counter(opponent.history)[C])/len(opponent.history)*100
    
    @staticmethod
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

    
    @staticmethod
    def calcforgiveness(self, opponent):
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

        # if DCounter > 0 and punishmentCounter < DCounter:
        #     return (100-(punishmentCounter/DCounter*100))
        # elif punishmentCounter >= DCounter:
        #     return 0
        # else:
        #     return 100                

        if punishmentCounter > 0:
            return DCounter/punishmentCounter*100
        else:
            return 100
    
    @staticmethod
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
    
    @staticmethod
    def printToFile(first_time, opponent, coop, adap, forg, stoch, output, played):
        if first_time is False:
            WriteToExcel.changeLastCell(opponent.history[-1])

        WriteToExcel.appendRows([len(opponent.history)+1, coop, adap, forg, stoch, output, played, ""])
    
    @staticmethod
    def sigmoid(x, center, scale=10):
        return 1 / (1 + exp(-scale * (x - center)))
    
    @staticmethod
    def fuzzy_gate(mu_D, mu_C, d_center = 0.4, c_center = 0.6):
        d_condition = FuzzyMethods.sigmoid(mu_D, center=d_center)
        c_condition = 1 - FuzzyMethods.sigmoid(mu_C, center=c_center)

        w1 = d_condition * c_condition
        w2 = 1 - w1

        z1 = 1
        z2 = 0

        z = (w1 * z1 + w2 * z2) / (w1 + w2 + 1e-6)
        return z
    
    
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
    forgiveness = ctrl.Antecedent(np.arange(0, 100, 1), 'forgiveness')
    stochastic = ctrl.Antecedent(np.arange(0, 100, 1), 'stochastic')

    cooperation.automf(names=["low", "medium", "high"])
    adaptivity.automf(names=["no", "yes"])
    forgiveness.automf(names=["low", "medium", "high"])
    forgiveness['low'] = fuzz.gaussmf(forgiveness.universe, 0, 25)
    forgiveness['medium'] = fuzz.trimf(forgiveness.universe, [25, 50, 75])
    stochastic.automf(names=["none", "sometimes", "always"])

    resulting_strategy = ctrl.Consequent(np.arange(0, 100, 1),'resulting_strategy')
    resulting_strategy['D'] = fuzz.trimf(resulting_strategy.universe, [0, 25, 50])
    resulting_strategy['C'] = fuzz.trimf(resulting_strategy.universe, [35, 75, 100])

    rule1 = ctrl.Rule(cooperation['high'] & adaptivity['no'] & (forgiveness['medium'] | forgiveness['high']), resulting_strategy['D'])
    rule2 = ctrl.Rule(forgiveness['low'] & cooperation['high'], resulting_strategy['C'])
    rule3 = ctrl.Rule(stochastic['always'] | adaptivity['no'], resulting_strategy['D'])
    rule4 = ctrl.Rule(cooperation['low'] | (cooperation['medium'] & forgiveness['low']), resulting_strategy['D'])
    rule5 = ctrl.Rule(cooperation['medium'] & forgiveness['medium'] & adaptivity['yes'], resulting_strategy['C'])

    strategy_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5])
    chosen_strategy = ctrl.ControlSystemSimulation(strategy_ctrl)

    h = {'Name':'', 'Fuzzy' : [], 'Opponent': []}

    forgiveness_index = 0
    stochastic_index = 0
    first_time = True

    
    def strategy(self, opponent: Player) -> Action:

        if(len(self.history) == 0 or D not in opponent.history):
            return C
        
        else:

            coop = FuzzyMethods.calcCooperation(self, opponent)
            adap = FuzzyMethods.calcAdaptivity(self, opponent)
            forg = FuzzyMethods.calcforgiveness(self, opponent)
            stoch = FuzzyMethods.calcStochastic(self, opponent)

            self.chosen_strategy.input['cooperation'] = coop
            self.chosen_strategy.input['adaptivity'] = adap
            self.chosen_strategy.input['forgiveness'] = forg
            self.chosen_strategy.input['stochastic'] = stoch
            
            self.chosen_strategy.compute()

            d_membership = fuzz.interp_membership(self.resulting_strategy.universe, self.resulting_strategy['D'].mf, self.chosen_strategy.output['resulting_strategy'])
            c_membership = fuzz.interp_membership(self.resulting_strategy.universe, self.resulting_strategy['C'].mf, self.chosen_strategy.output['resulting_strategy'])

            # output = "C"

            # if(d_membership >= 0.4 and c_membership < 0.6):
            #     output = "D"

            # FuzzyMethods.printToFile(self.first_time, opponent, coop, adap, forg, stoch, output, output)
           
            # if self.first_time == True:
            #     self.first_time = False

            if(d_membership >= 0.4 and c_membership < 0.6):
                return D
        
        return C

class FuzzySugeno(Player):

    # These are various properties for the strategy
    name = "FuzzySugeno"
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
    forgiveness = ctrl.Antecedent(np.arange(0, 100, 1), 'forgiveness')
    stochastic = ctrl.Antecedent(np.arange(0, 100, 1), 'stochastic')

    cooperation.automf(names=["low", "medium", "high"])
    adaptivity.automf(names=["no", "yes"])
    forgiveness.automf(names=["low", "medium", "high"])
    forgiveness['low'] = fuzz.gaussmf(forgiveness.universe, 0, 25)
    forgiveness['medium'] = fuzz.trimf(forgiveness.universe, [25, 50, 75])
    stochastic.automf(names=["none", "sometimes", "always"])

    resulting_strategy = ctrl.Consequent(np.arange(0, 100, 1),'resulting_strategy')
    resulting_strategy['D'] = fuzz.trimf(resulting_strategy.universe, [0, 25, 50])
    resulting_strategy['C'] = fuzz.trimf(resulting_strategy.universe, [35, 75, 100])

    rule1 = ctrl.Rule(cooperation['high'] & adaptivity['no'] & (forgiveness['medium'] | forgiveness['high']), resulting_strategy['D'])
    rule2 = ctrl.Rule(forgiveness['low'] & cooperation['high'], resulting_strategy['C'])
    rule3 = ctrl.Rule(stochastic['always'] | adaptivity['no'], resulting_strategy['D'])
    rule4 = ctrl.Rule(cooperation['low'] | (cooperation['medium'] & forgiveness['low']), resulting_strategy['D'])
    rule5 = ctrl.Rule(cooperation['medium'] & forgiveness['medium'] & adaptivity['yes'], resulting_strategy['C'])

    strategy_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5])
    chosen_strategy = ctrl.ControlSystemSimulation(strategy_ctrl)

    h = {'Name':'', 'Fuzzy' : [], 'Opponent': []}

    forgiveness_index = 0
    stochastic_index = 0
    first_time = True
 
    def strategy(self, opponent: Player) -> Action:

        if(len(self.history) == 0 or D not in opponent.history):
            return C
        
        else:
            coop = FuzzyMethods.calcCooperation(self, opponent)
            adap = FuzzyMethods.calcAdaptivity(self, opponent)
            forg = FuzzyMethods.calcforgiveness(self, opponent)
            stoch = FuzzyMethods.calcStochastic(self, opponent)

            self.chosen_strategy.input['cooperation'] = coop
            self.chosen_strategy.input['adaptivity'] = adap
            self.chosen_strategy.input['forgiveness'] = forg
            self.chosen_strategy.input['stochastic'] = stoch
            
            self.chosen_strategy.compute()

            d_membership = fuzz.interp_membership(self.resulting_strategy.universe, self.resulting_strategy['D'].mf, self.chosen_strategy.output['resulting_strategy'])
            c_membership = fuzz.interp_membership(self.resulting_strategy.universe, self.resulting_strategy['C'].mf, self.chosen_strategy.output['resulting_strategy'])

            output = FuzzyMethods.fuzzy_gate(d_membership, c_membership)

            # played = "D" if output > 0.5 else "C"

            # FuzzyMethods.printToFile(self.first_time, opponent, coop, adap, forg, stoch, output, played)

            # if self.first_time == True:
            #     self.first_time = False

            if(output > 0.5):
                return D
        
        return C

class OptunaFuzzy(Player):

    # These are various properties for the strategy
    name = "OptunaFuzzyMamdani"
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
    forgiveness = ctrl.Antecedent(np.arange(0, 100, 1), 'forgiveness')
    stochastic = ctrl.Antecedent(np.arange(0, 100, 1), 'stochastic')

    cooperation.automf(names=["low", "medium", "high"])
    adaptivity.automf(names=["no", "yes"])
    forgiveness.automf(names=["low", "medium", "high"])
    forgiveness['low'] = fuzz.gaussmf(forgiveness.universe, 0, 25)
    forgiveness['medium'] = fuzz.trimf(forgiveness.universe, [25, 50, 75])
    stochastic.automf(names=["none", "sometimes", "always"])

    resulting_strategy = ctrl.Consequent(np.arange(0, 100, 1),'resulting_strategy')
    resulting_strategy['D'] = fuzz.trimf(resulting_strategy.universe, [0, 8, 9])
    resulting_strategy['C'] = fuzz.trimf(resulting_strategy.universe, [59, 60, 99])

    rule1 = ctrl.Rule(cooperation['high'] & adaptivity['no'] & (forgiveness['medium'] | forgiveness['high']), resulting_strategy['D'])
    rule2 = ctrl.Rule(forgiveness['low'] & cooperation['high'], resulting_strategy['C'])
    rule3 = ctrl.Rule(stochastic['always'] | adaptivity['no'], resulting_strategy['D'])
    rule4 = ctrl.Rule(cooperation['low'] | (cooperation['medium'] & forgiveness['low']), resulting_strategy['D'])
    rule5 = ctrl.Rule(cooperation['medium'] & forgiveness['medium'] & adaptivity['yes'], resulting_strategy['C'])

    strategy_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5])
    chosen_strategy = ctrl.ControlSystemSimulation(strategy_ctrl)

    h = {'Name':'', 'Fuzzy' : [], 'Opponent': []}

    forgiveness_index = 0
    stochastic_index = 0
    first_time = True

    
    def strategy(self, opponent: Player) -> Action:

        if(len(self.history) == 0 or D not in opponent.history):
            return C
        
        else:

            coop = FuzzyMethods.calcCooperation(self, opponent)
            adap = FuzzyMethods.calcAdaptivity(self, opponent)
            forg = FuzzyMethods.calcforgiveness(self, opponent)
            stoch = FuzzyMethods.calcStochastic(self, opponent)

            self.chosen_strategy.input['cooperation'] = coop
            self.chosen_strategy.input['adaptivity'] = adap
            self.chosen_strategy.input['forgiveness'] = forg
            self.chosen_strategy.input['stochastic'] = stoch
            
            self.chosen_strategy.compute()

            d_membership = fuzz.interp_membership(self.resulting_strategy.universe, self.resulting_strategy['D'].mf, self.chosen_strategy.output['resulting_strategy'])
            c_membership = fuzz.interp_membership(self.resulting_strategy.universe, self.resulting_strategy['C'].mf, self.chosen_strategy.output['resulting_strategy'])

            # output = "C"

            # if(d_membership >= 0.4 and c_membership < 0.6):
            #     output = "D"

            # FuzzyMethods.printToFile(self.first_time, opponent, coop, adap, forg, stoch, output, output)
           
            # if self.first_time == True:
            #     self.first_time = False

            if(d_membership >= 0.35466 and c_membership < 0.28499):
                return D
        
        return C

class OptunaFuzzySugeno(Player):

    # These are various properties for the strategy
    name = "OptunaFuzzySugeno"
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
    forgiveness = ctrl.Antecedent(np.arange(0, 100, 1), 'forgiveness')
    stochastic = ctrl.Antecedent(np.arange(0, 100, 1), 'stochastic')

    cooperation.automf(names=["low", "medium", "high"])
    adaptivity.automf(names=["no", "yes"])
    forgiveness.automf(names=["low", "medium", "high"])
    forgiveness['low'] = fuzz.gaussmf(forgiveness.universe, 0, 25)
    forgiveness['medium'] = fuzz.trimf(forgiveness.universe, [25, 50, 75])
    stochastic.automf(names=["none", "sometimes", "always"])

    resulting_strategy = ctrl.Consequent(np.arange(0, 100, 1),'resulting_strategy')
    resulting_strategy['D'] = fuzz.trimf(resulting_strategy.universe, [0, 26, 28])
    resulting_strategy['C'] = fuzz.trimf(resulting_strategy.universe, [41, 66, 99])

    rule1 = ctrl.Rule(cooperation['high'] & adaptivity['no'] & (forgiveness['medium'] | forgiveness['high']), resulting_strategy['D'])
    rule2 = ctrl.Rule(forgiveness['low'] & cooperation['high'], resulting_strategy['C'])
    rule3 = ctrl.Rule(stochastic['always'] | adaptivity['no'], resulting_strategy['D'])
    rule4 = ctrl.Rule(cooperation['low'] | (cooperation['medium'] & forgiveness['low']), resulting_strategy['D'])
    rule5 = ctrl.Rule(cooperation['medium'] & forgiveness['medium'] & adaptivity['yes'], resulting_strategy['C'])

    strategy_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5])
    chosen_strategy = ctrl.ControlSystemSimulation(strategy_ctrl)

    h = {'Name':'', 'Fuzzy' : [], 'Opponent': []}

    forgiveness_index = 0
    stochastic_index = 0
    first_time = True
 
    def strategy(self, opponent: Player) -> Action:

        if(len(self.history) == 0 or D not in opponent.history):
            return C
        
        else:
            coop = FuzzyMethods.calcCooperation(self, opponent)
            adap = FuzzyMethods.calcAdaptivity(self, opponent)
            forg = FuzzyMethods.calcforgiveness(self, opponent)
            stoch = FuzzyMethods.calcStochastic(self, opponent)

            self.chosen_strategy.input['cooperation'] = coop
            self.chosen_strategy.input['adaptivity'] = adap
            self.chosen_strategy.input['forgiveness'] = forg
            self.chosen_strategy.input['stochastic'] = stoch
            
            self.chosen_strategy.compute()

            d_membership = fuzz.interp_membership(self.resulting_strategy.universe, self.resulting_strategy['D'].mf, self.chosen_strategy.output['resulting_strategy'])
            c_membership = fuzz.interp_membership(self.resulting_strategy.universe, self.resulting_strategy['C'].mf, self.chosen_strategy.output['resulting_strategy'])

            output = FuzzyMethods.fuzzy_gate(d_membership, c_membership, 0.32685, 0.52012)

            if(output > 0.5):
                return D
        
        return C

class PSOFuzzySugeno(Player):

    # These are various properties for the strategy
    name = "PSOFuzzySugeno"
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
    forgiveness = ctrl.Antecedent(np.arange(0, 100, 1), 'forgiveness')
    stochastic = ctrl.Antecedent(np.arange(0, 100, 1), 'stochastic')

    cooperation.automf(names=["low", "medium", "high"])
    adaptivity.automf(names=["no", "yes"])
    forgiveness.automf(names=["low", "medium", "high"])
    forgiveness['low'] = fuzz.gaussmf(forgiveness.universe, 0, 25)
    forgiveness['medium'] = fuzz.trimf(forgiveness.universe, [25, 50, 75])
    stochastic.automf(names=["none", "sometimes", "always"])

    resulting_strategy = ctrl.Consequent(np.arange(0, 100, 1),'resulting_strategy')
    resulting_strategy['D'] = fuzz.trimf(resulting_strategy.universe, [0, 22.887, 40.3935])
    resulting_strategy['C'] = fuzz.trimf(resulting_strategy.universe, [64.26314, 65.8584, 99])

    rule1 = ctrl.Rule(cooperation['high'] & adaptivity['no'] & (forgiveness['medium'] | forgiveness['high']), resulting_strategy['D'])
    rule2 = ctrl.Rule(forgiveness['low'] & cooperation['high'], resulting_strategy['C'])
    rule3 = ctrl.Rule(stochastic['always'] | adaptivity['no'], resulting_strategy['D'])
    rule4 = ctrl.Rule(cooperation['low'] | (cooperation['medium'] & forgiveness['low']), resulting_strategy['D'])
    rule5 = ctrl.Rule(cooperation['medium'] & forgiveness['medium'] & adaptivity['yes'], resulting_strategy['C'])

    strategy_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5])
    chosen_strategy = ctrl.ControlSystemSimulation(strategy_ctrl)

    h = {'Name':'', 'Fuzzy' : [], 'Opponent': []}

    forgiveness_index = 0
    stochastic_index = 0
    first_time = True
 
    def strategy(self, opponent: Player) -> Action:

        if(len(self.history) == 0 or D not in opponent.history):
            return C
        
        else:
            coop = FuzzyMethods.calcCooperation(self, opponent)
            adap = FuzzyMethods.calcAdaptivity(self, opponent)
            forg = FuzzyMethods.calcforgiveness(self, opponent)
            stoch = FuzzyMethods.calcStochastic(self, opponent)

            self.chosen_strategy.input['cooperation'] = coop
            self.chosen_strategy.input['adaptivity'] = adap
            self.chosen_strategy.input['forgiveness'] = forg
            self.chosen_strategy.input['stochastic'] = stoch
            
            self.chosen_strategy.compute()

            d_membership = fuzz.interp_membership(self.resulting_strategy.universe, self.resulting_strategy['D'].mf, self.chosen_strategy.output['resulting_strategy'])
            c_membership = fuzz.interp_membership(self.resulting_strategy.universe, self.resulting_strategy['C'].mf, self.chosen_strategy.output['resulting_strategy'])

            output = FuzzyMethods.fuzzy_gate(d_membership, c_membership, 0.42124, 0.67085)

            if(output > 0.5):
                return D
        
        return C

class PravilaFuzzy(Player):

    # These are various properties for the strategy
    name = "PravilaFuzzy"
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
    forgiveness = ctrl.Antecedent(np.arange(0, 100, 1), 'forgiveness')
    stochastic = ctrl.Antecedent(np.arange(0, 100, 1), 'stochastic')

    cooperation.automf(names=["low", "medium", "high"])
    adaptivity.automf(names=["no", "yes"])
    forgiveness.automf(names=["low", "medium", "high"])
    forgiveness['low'] = fuzz.gaussmf(forgiveness.universe, 0, 25)
    forgiveness['medium'] = fuzz.trimf(forgiveness.universe, [25, 50, 75])
    stochastic.automf(names=["none", "sometimes", "always"])

    resulting_strategy = ctrl.Consequent(np.arange(0, 100, 1),'resulting_strategy')
    resulting_strategy['D'] = fuzz.trimf(resulting_strategy.universe, [0, 25, 50])
    resulting_strategy['C'] = fuzz.trimf(resulting_strategy.universe, [35, 75, 100])

    rule1 = ctrl.Rule(cooperation['medium'] & forgiveness['high'] & stochastic['always'], resulting_strategy['C'])
    rule2 = ctrl.Rule(cooperation['high'] & forgiveness['medium'] & stochastic['sometimes'], resulting_strategy['D'])
    rule3 = ctrl.Rule( adaptivity['yes'] & cooperation['medium'] & stochastic['always'], resulting_strategy['D'])
    
    strategy_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
    chosen_strategy = ctrl.ControlSystemSimulation(strategy_ctrl)

    h = {'Name':'', 'Fuzzy' : [], 'Opponent': []}

    forgiveness_index = 0
    stochastic_index = 0
    first_time = True

    
    def strategy(self, opponent: Player) -> Action:

        if(len(self.history) == 0 or D not in opponent.history):
            return C
        
        else:

            coop = FuzzyMethods.calcCooperation(self, opponent)
            adap = FuzzyMethods.calcAdaptivity(self, opponent)
            forg = FuzzyMethods.calcforgiveness(self, opponent)
            stoch = FuzzyMethods.calcStochastic(self, opponent)

            self.chosen_strategy.input['cooperation'] = coop
            self.chosen_strategy.input['adaptivity'] = adap
            self.chosen_strategy.input['forgiveness'] = forg
            self.chosen_strategy.input['stochastic'] = stoch
            
            try:

                self.chosen_strategy.compute()

                d_membership = fuzz.interp_membership(self.resulting_strategy.universe, self.resulting_strategy['D'].mf, self.chosen_strategy.output['resulting_strategy'])
                c_membership = fuzz.interp_membership(self.resulting_strategy.universe, self.resulting_strategy['C'].mf, self.chosen_strategy.output['resulting_strategy'])

                if(d_membership >= 0.4 and c_membership < 0.6):
                    return D
            except:
                return C
        
        return C

class Pravila2Fuzzy(Player):

    # These are various properties for the strategy
    name = "Pravila2Fuzzy"
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
    forgiveness = ctrl.Antecedent(np.arange(0, 100, 1), 'forgiveness')
    stochastic = ctrl.Antecedent(np.arange(0, 100, 1), 'stochastic')

    cooperation.automf(names=["low", "medium", "high"])
    adaptivity.automf(names=["no", "yes"])
    forgiveness.automf(names=["low", "medium", "high"])
    forgiveness['low'] = fuzz.gaussmf(forgiveness.universe, 0, 25)
    forgiveness['medium'] = fuzz.trimf(forgiveness.universe, [25, 50, 75])
    stochastic.automf(names=["none", "sometimes", "always"])

    resulting_strategy = ctrl.Consequent(np.arange(0, 100, 1),'resulting_strategy')
    resulting_strategy['D'] = fuzz.trimf(resulting_strategy.universe, [0, 25, 50])
    resulting_strategy['C'] = fuzz.trimf(resulting_strategy.universe, [35, 75, 100])

    rule1 = ctrl.Rule(cooperation['medium'] & stochastic['none'], resulting_strategy['C'])
    rule2 = ctrl.Rule(cooperation['medium'] & forgiveness['low'] & stochastic['always'], resulting_strategy['C'])
    rule3 = ctrl.Rule(adaptivity['yes'] & cooperation['medium'] & forgiveness['medium'] & stochastic['always'], resulting_strategy['D'])
    rule4 = ctrl.Rule(adaptivity['yes'] & cooperation['high'] & forgiveness['medium'] & stochastic['always'], resulting_strategy['D'])

    strategy_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4])
    chosen_strategy = ctrl.ControlSystemSimulation(strategy_ctrl)

    h = {'Name':'', 'Fuzzy' : [], 'Opponent': []}

    forgiveness_index = 0
    stochastic_index = 0
    first_time = True

    
    def strategy(self, opponent: Player) -> Action:

        if(len(self.history) == 0 or D not in opponent.history):
            return C
        
        else:

            coop = FuzzyMethods.calcCooperation(self, opponent)
            adap = FuzzyMethods.calcAdaptivity(self, opponent)
            forg = FuzzyMethods.calcforgiveness(self, opponent)
            stoch = FuzzyMethods.calcStochastic(self, opponent)

            self.chosen_strategy.input['cooperation'] = coop
            self.chosen_strategy.input['adaptivity'] = adap
            self.chosen_strategy.input['forgiveness'] = forg
            self.chosen_strategy.input['stochastic'] = stoch
            
            try:
                self.chosen_strategy.compute()
                if self.chosen_strategy.output['resulting_strategy'] > 50:
                    return D
            except:
                return C
            return C
        
        return C
