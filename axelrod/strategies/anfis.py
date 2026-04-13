
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
    def sigmoid(x, center, scale=10):
        return 1 / (1 + exp(-scale * (x - center)))
    
    @staticmethod
    def fuzzy_gate(mu_D, mu_C):
        d_condition = FuzzyMethods.sigmoid(mu_D, center=0.4)
        c_condition = 1 - FuzzyMethods.sigmoid(mu_C, center=0.6)

        w1 = d_condition * c_condition
        w2 = 1 - w1

        z1 = 1
        z2 = 0

        z = (w1 * z1 + w2 * z2) / (w1 + w2 + 1e-6)
        return z
  
class GaussianMF(torch.nn.Module):
    def __init__(self, c, sigma):
        super().__init__()
        self.c = torch.nn.Parameter(torch.tensor(float(c)))
        self.sigma = torch.nn.Parameter(torch.tensor(float(sigma)))
    def forward(self, x):
        return torch.exp(-0.5 * ((x - self.c) / self.sigma)**2)

class ANFIS(torch.nn.Module):
    def __init__(self, n_inputs, mfs_per_input):
        super().__init__()
        self.n_inputs = n_inputs
        self.m = mfs_per_input
        self.mf_layer = torch.nn.ModuleList([
            torch.nn.ModuleList([GaussianMF(0.5, 0.2) for _ in range(mfs_per_input)])
            for _ in range(n_inputs)
        ])
        self.n_rules = mfs_per_input ** n_inputs
        self.linear = torch.nn.Linear(n_inputs, self.n_rules, bias=True)
    def forward(self, x):
        batch_size = x.size(0)
        memberships = torch.stack([
            torch.stack([mf(x[:, i]) for mf in self.mf_layer[i]], dim=-1)
            for i in range(self.n_inputs)
        ], dim=1)
        rule_strengths = memberships[:,0,:]
        for i in range(1, self.n_inputs):
            rule_strengths = rule_strengths.unsqueeze(2) * memberships[:,i,:].unsqueeze(1)
            rule_strengths = rule_strengths.reshape(batch_size, -1)
        w_normalized = rule_strengths / (rule_strengths.sum(dim=1, keepdim=True) + 1e-6)
        linear_output = self.linear(x)
        output = (w_normalized * linear_output).sum(dim=1, keepdim=True)
        return output

class ANFISStrategy(Player):
    name = "ANFIS"
    classifier = {
        "memory_depth": float("inf"),
        "stochastic": False,
        "long_run_time": False,
        "inspects_source": False,
        "manipulates_source": False,
        "manipulates_state": False,
    }

    # 1. Inicijalizuj model
    model_loaded = ANFIS(n_inputs=4, mfs_per_input=3)

    # 2. Učitaj parametre
    model_loaded.load_state_dict(torch.load(r"C:\Users\Ognjen\axl_mab\Axelrod\mab\anfis3\anfis_model.pt"))
    model_loaded.eval()

    # 3. Učitaj skalere
    scaler_X = joblib.load("scaler_X.save")
    scaler_y = joblib.load("scaler_y.save")

    def predict_anfis(self, X_input):
        """
        X_input : np.array shape [n_samples, 4] (Coop, Adap, Forg, Stoch)
        vraća: np.array shape [n_samples, 1] sa predikcijom Output
        """

        Xn = self.scaler_X.transform(X_input)
        X_tensor = torch.tensor(Xn, dtype=torch.float32)

        with torch.no_grad():
            y_pred = self.model_loaded(X_tensor).numpy()

        y_pred_rescaled = self.scaler_y.inverse_transform(y_pred)
        return y_pred_rescaled
    
    def strategy(self, opponent: Player) -> Action:

        if(len(self.history) == 0 or D not in opponent.history):
            return C
        else:
            prediction = self.predict_anfis(np.array(
                [
                    FuzzyMethods.calcCooperation(self, opponent),
                    FuzzyMethods.calcAdaptivity(self, opponent),
                    FuzzyMethods.calcforgiveness(self, opponent),
                    FuzzyMethods.calcStochastic(self, opponent)
                ]).reshape(1, -1))
            # print("Predikcija ANFIS:", prediction)
            if(prediction > 0.5):
                return D
            return C

        

   
            
