import pandas as pd

df1 = pd.read_csv("hand_data_rock.csv")
df2 = pd.read_csv("hand_data_paper.csv")
df3 = pd.read_csv("hand_data_scissors.csv")


df_all = pd.concat([df1, df2, df3])
df_all.to_csv("hand_gesture_data.csv", index=False)
