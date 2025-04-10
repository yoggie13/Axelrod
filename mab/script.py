import axelrod as axl
import pandas as pd
from datetime import datetime

p = [player() for player in axl.fuzzy_strategies]
# p = [player() for player in axl.ognjen_strategies]

# for i in range(1, 2):
t = axl.Tournament(players=p, repetitions=1)

results = t.play()
df = pd.DataFrame(
        {
            "Name": [x.name for x in p],
            "#Wins": list(map(lambda x: sum(x), results.wins)),
            "TotalScore": list(map(lambda x: sum(x), results.scores)),
            "#Cooperations" : list(map(lambda x: sum(x), results.cooperation)),
            "CooperationRating" : results.cooperating_rating
        })

# dfPayoffMatrix = pd.DataFrame(results.payoff_matrix, columns=[x.name for x in p], index = [x.name for x in p])
# dfCooperationMatrix = pd.DataFrame(results.cooperation, columns=[x.name for x in p], index = [x.name for x in p])
# dfNormalizedCooperationMatrix = pd.DataFrame(results.normalised_cooperation, columns=[x.name for x in p], index = [x.name for x in p])


# # df['TotalWins'] = df[["Wins1", "Wins2", "Wins3", "Wins4","Wins5"]].sum(axis=1)
# # df['TotalScores'] = df[["Scores1", "Scores2", "Scores3", "Scores4","Scores5"]].sum(axis=1)

# # print(df.sort_values("TotalScores", ascending=False))
# df = df.sort_values("TotalScore", ascending=False)
# df = df.reset_index(drop=True)


# with pd.ExcelWriter(f"output_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.xlsx") as writer:
#     df.to_excel(writer, sheet_name="Rank_List", startrow=0)
#     dfPayoffMatrix.to_excel(writer, sheet_name="Payoff_Matrix", startrow=0)
#     dfCooperationMatrix.to_excel(writer, sheet_name="Cooperation_Matrix", startrow=0)
#     dfNormalizedCooperationMatrix.to_excel(writer, sheet_name="Normalized_Cooperation_Matrix", startrow=0)

