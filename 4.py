#4
import pandas as pd

def find_s_algorithm(file_path):
    data = pd.read_csv(file_path)

    print("Training Data:")
    print(data)

    attributes = data.columns[:-1]
    target = data.columns[-1]

    hypothesis = None

    for _, row in data.iterrows():
        if row[target] == "Yes":
            if hypothesis is None:
                hypothesis = list(row[attributes])
            else:
                for i in range(len(hypothesis)):
                    if hypothesis[i] != row[attributes[i]]:
                        hypothesis[i] = '?'

    print("\nFinal Hypothesis:")
    print(hypothesis)

find_s_algorithm("fan_swings.csv")
# Switch_Status,Fan_Working,Fan_Swings
# On,Yes,Yes
# On,No,No
# Off,Yes,No
# Off,No,No
