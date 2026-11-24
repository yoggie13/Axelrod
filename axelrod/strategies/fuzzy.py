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
    def fuzzy_gate(mu_D, mu_C, d_center, c_center):
        d_condition = FuzzyMethods.sigmoid(mu_D, center=d_center)
        c_condition = 1 - FuzzyMethods.sigmoid(mu_C, center=c_center)

        w1 = d_condition * c_condition
        
        return w1
     
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

class FuzzyMamdaniGA(Player):

    name = "FuzzyMamdaniGA"
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
    resulting_strategy['D'] = fuzz.trimf(resulting_strategy.universe, [0, 63, 86])
    resulting_strategy['C'] = fuzz.trimf(resulting_strategy.universe, [34, 76, 99 ])

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
            
            try:
                self.chosen_strategy.compute()
                output = self.chosen_strategy.output['resulting_strategy']
            except:
                print("Ulazi:", self.chosen_strategy.input)
                raise

            if output > 50:
                return C
            else:
                return D
        
        return C

class FuzzyMamdaniO(Player):

    name = "FuzzyMamdaniO"
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
    resulting_strategy['D'] = fuzz.trimf(resulting_strategy.universe, [22, 53, 68])
    resulting_strategy['C'] = fuzz.trimf(resulting_strategy.universe, [19, 81, 98])

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

            output = self.chosen_strategy.output['resulting_strategy']

            if output > 50:
                return C
            else:
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

            output = FuzzyMethods.fuzzy_gate(d_membership, c_membership, 0.4, 0.6)

            # played = "D" if output > 0.5 else "C"

            # FuzzyMethods.printToFile(self.first_time, opponent, coop, adap, forg, stoch, output, played)

            # if self.first_time == True:
            #     self.first_time = False

            if(output > 0.5):
                return D
        
        return C

class FSGA1(Player):

    name = "FuzzySugenoGA1"
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

            output = FuzzyMethods.fuzzy_gate(d_membership, 0.91984766 , c_membership, 0.98164158)

            if(output > 0.5):
                return D
            
        return C

class FSGA2(Player):

    name = "FuzzySugenoGA2"
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

            output = FuzzyMethods.fuzzy_gate(d_membership, 0.77799452, c_membership, 0.76518589)

            if(output > 0.5):
                return D
            
        return C

class FSGA3(Player):

    name = "FuzzySugenoGA3"
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
    resulting_strategy['D'] = fuzz.trimf(resulting_strategy.universe, [0, 20, 41])
    resulting_strategy['C'] = fuzz.trimf(resulting_strategy.universe, [9, 92, 99])

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

            output = FuzzyMethods.fuzzy_gate(d_membership, 0.5, c_membership, 0.5)

            if(output > 0.5):
                return D
            
        return C

class FSO1(Player):

    name = "FSO1"
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

            output = FuzzyMethods.fuzzy_gate(d_membership, 0.7593470762343023, c_membership, 0.5192838614249861)

            if(output > 0.5):
                return D
            
        return C

class FSO2(Player):

    name = "FSO2"
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

            output = FuzzyMethods.fuzzy_gate(d_membership, 0.7334935854745204, c_membership, 0.7429117868191882)

            if(output > 0.5):
                return D
            
        return C

class FSO3(Player):

    name = "FuzzySugenoO3"
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
    resulting_strategy['D'] = fuzz.trimf(resulting_strategy.universe, [90, 96, 98])
    resulting_strategy['C'] = fuzz.trimf(resulting_strategy.universe, [57, 81, 94])

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

            output = FuzzyMethods.fuzzy_gate(d_membership, 0.5, c_membership, 0.5)

            if(output > 0.5):
                return D
            
        return C

class FuzzyMamdaniRules8(Player):

    name = "FuzzyMamdaniRules8"
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

    paramsMatrix = np.array([3, 2, 0, 2, 1, 3, 0, 3, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 3, 1, 1, 2, 0, 0, 1, 1, 0, 2, 3, 0, 2, 1, 1, 0, 1, 0, 2, 3, 1, 0]).reshape(-1, 5)

    rules = []

    for ruleVector in paramsMatrix:
        coop = None  if ruleVector[0] == 0 else "low" if ruleVector[0] == 1 else "medium" if ruleVector[0] == 2 else "high"
        adap = None if ruleVector[1] == 0 else "no" if ruleVector[1] == 1 else "yes"
        forg = None if ruleVector[2] == 0 else "low" if ruleVector[2] == 1 else "medium" if ruleVector[2] == 2  else "high"
        stoch = None if ruleVector[3] == 0 else "none" if ruleVector[3] == 1 else "sometimes" if ruleVector[3] == 2 else "always"
        res = "D" if ruleVector[4] == 0 else "C"

        antecedents = []

        if coop is not None:
            antecedents.append(cooperation[coop])

        if adap is not None:
            antecedents.append(adaptivity[adap])

        if forg is not None:
            antecedents.append(forgiveness[forg])

        if stoch is not None:
            antecedents.append(stochastic[stoch])

        if len(antecedents) == 0:
            continue 

        rule_lhs = antecedents[0]
        
        for a in antecedents[1:]:
            rule_lhs = rule_lhs & a

        rule = ctrl.Rule(rule_lhs, resulting_strategy[res])
        rules.append(rule)

    strategy_ctrl = ctrl.ControlSystem(rules)
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

            try:
                output = self.chosen_strategy.output['resulting_strategy']
            except Exception:
                print("Error computing output for FuzzyMamdaniRules5")
                print(f"Inputs: cooperation={coop}, adaptivity={adap}, forgiveness={forg}, stochastic={stoch}")
                return C
            
            if output > 50:
                return C
            else:
                return D
        
        return C

class FuzzyMamdaniRules7(Player):

    name = "FuzzyMamdaniRules7"
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

    paramsMatrix = np.array([3, 2, 0, 2, 1, 3, 0, 3, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 3, 1, 1, 2, 0, 0, 1, 2, 1, 1, 0, 1, 0, 2, 3, 1, 0]).reshape(-1, 5)

    rules = []

    for ruleVector in paramsMatrix:
        coop = None  if ruleVector[0] == 0 else "low" if ruleVector[0] == 1 else "medium" if ruleVector[0] == 2 else "high"
        adap = None if ruleVector[1] == 0 else "no" if ruleVector[1] == 1 else "yes"
        forg = None if ruleVector[2] == 0 else "low" if ruleVector[2] == 1 else "medium" if ruleVector[2] == 2  else "high"
        stoch = None if ruleVector[3] == 0 else "none" if ruleVector[3] == 1 else "sometimes" if ruleVector[3] == 2 else "always"
        res = "D" if ruleVector[4] == 0 else "C"

        antecedents = []

        if coop is not None:
            antecedents.append(cooperation[coop])

        if adap is not None:
            antecedents.append(adaptivity[adap])

        if forg is not None:
            antecedents.append(forgiveness[forg])

        if stoch is not None:
            antecedents.append(stochastic[stoch])

        if len(antecedents) == 0:
            continue 

        rule_lhs = antecedents[0]
        
        for a in antecedents[1:]:
            rule_lhs = rule_lhs & a

        rule = ctrl.Rule(rule_lhs, resulting_strategy[res])
        rules.append(rule)

    strategy_ctrl = ctrl.ControlSystem(rules)
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

            try:
                output = self.chosen_strategy.output['resulting_strategy']
            except Exception:
                print("Error computing output for FuzzyMamdaniRules5")
                print(f"Inputs: cooperation={coop}, adaptivity={adap}, forgiveness={forg}, stochastic={stoch}")
                return C
            
            if output > 50:
                return C
            else:
                return D
        
        return C

class FuzzyMamdaniRules6(Player):

    name = "FuzzyMamdaniRules6"
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

    paramsMatrix = np.array([3, 0, 0, 1, 1, 3, 0, 3, 0, 0, 1, 0, 1, 0, 0, 1, 2, 0, 0, 1, 2, 1, 2, 0, 1, 0, 0, 2, 1, 1]).reshape(-1, 5)

    rules = []

    for ruleVector in paramsMatrix:
        coop = None  if ruleVector[0] == 0 else "low" if ruleVector[0] == 1 else "medium" if ruleVector[0] == 2 else "high"
        adap = None if ruleVector[1] == 0 else "no" if ruleVector[1] == 1 else "yes"
        forg = None if ruleVector[2] == 0 else "low" if ruleVector[2] == 1 else "medium" if ruleVector[2] == 2  else "high"
        stoch = None if ruleVector[3] == 0 else "none" if ruleVector[3] == 1 else "sometimes" if ruleVector[3] == 2 else "always"
        res = "D" if ruleVector[4] == 0 else "C"

        antecedents = []

        if coop is not None:
            antecedents.append(cooperation[coop])

        if adap is not None:
            antecedents.append(adaptivity[adap])

        if forg is not None:
            antecedents.append(forgiveness[forg])

        if stoch is not None:
            antecedents.append(stochastic[stoch])

        if len(antecedents) == 0:
            continue 

        rule_lhs = antecedents[0]
        
        for a in antecedents[1:]:
            rule_lhs = rule_lhs & a

        rule = ctrl.Rule(rule_lhs, resulting_strategy[res])
        rules.append(rule)

    strategy_ctrl = ctrl.ControlSystem(rules)
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

            try:
                output = self.chosen_strategy.output['resulting_strategy']
            except Exception:
                print("Error computing output for FuzzyMamdaniRules5")
                print(f"Inputs: cooperation={coop}, adaptivity={adap}, forgiveness={forg}, stochastic={stoch}")
                return C
            
            if output > 50:
                return C
            else:
                return D
        
        return C

class FuzzyMamdaniRules5(Player):

    name = "FuzzyMamdaniRules5"
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

    paramsMatrix = np.array([3, 2, 0, 2, 1, 1, 0, 1, 0, 0, 1, 2, 0, 0, 1, 2, 1, 1, 0, 1, 0, 2, 3, 1, 0]).reshape(-1, 5)

    rules = []

    for ruleVector in paramsMatrix:
        coop = None  if ruleVector[0] == 0 else "low" if ruleVector[0] == 1 else "medium" if ruleVector[0] == 2 else "high"
        adap = None if ruleVector[1] == 0 else "no" if ruleVector[1] == 1 else "yes"
        forg = None if ruleVector[2] == 0 else "low" if ruleVector[2] == 1 else "medium" if ruleVector[2] == 2  else "high"
        stoch = None if ruleVector[3] == 0 else "none" if ruleVector[3] == 1 else "sometimes" if ruleVector[3] == 2 else "always"
        res = "D" if ruleVector[4] == 0 else "C"

        antecedents = []

        if coop is not None:
            antecedents.append(cooperation[coop])

        if adap is not None:
            antecedents.append(adaptivity[adap])

        if forg is not None:
            antecedents.append(forgiveness[forg])

        if stoch is not None:
            antecedents.append(stochastic[stoch])

        if len(antecedents) == 0:
            continue 

        rule_lhs = antecedents[0]
        
        for a in antecedents[1:]:
            rule_lhs = rule_lhs & a

        rule = ctrl.Rule(rule_lhs, resulting_strategy[res])
        rules.append(rule)

    strategy_ctrl = ctrl.ControlSystem(rules)
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

            try:
                output = self.chosen_strategy.output['resulting_strategy']
            except Exception:
                print("Error computing output for FuzzyMamdaniRules5")
                print(f"Inputs: cooperation={coop}, adaptivity={adap}, forgiveness={forg}, stochastic={stoch}")
                return C
            
            if output > 50:
                return C
            else:
                return D
        
        return C
