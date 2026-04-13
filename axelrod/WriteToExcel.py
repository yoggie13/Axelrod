import pandas as pd
from datetime import datetime

rows = []

def changeLastCell(value):
    global rows

    rows[-1][-1] = value


def appendRows(new_row):
    global rows
    rows.append(new_row)

def write_to_excel():
    df = pd.DataFrame(rows, columns=["ID", "Coop", "Adap", "Forg", "Stoch", "Output", "Played", "OpponentPlayed"])
    with pd.ExcelWriter(f"./output_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.xlsx") as writer:
        df.to_excel(writer, sheet_name="Rank_List", startrow=0)