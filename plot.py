

import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
data = pd.read_csv("ai4i2020.csv")

size_limit=10
# Extract the desired columns
air_temp = data.iloc[:, 3][:size_limit]  # 4th column
# col_5 = data.iloc[:, 4]
# col_6 = data.iloc[:, 5]
# col_7 = data.iloc[:, 6]
# col_8 = data.iloc[:, 7][:78]
col_9 = data.iloc[:, 8][:size_limit]




# Plotting
plt.figure(figsize=(10, 6))
plt.plot(air_temp, label='Air Temperature')
# plt.plot(col_5, label='process temp')
# plt.plot(col_6, label='rpm')
# plt.plot(col_7, label='torque')
# plt.plot(col_8, label='total wear')
plt.plot(list(map(lambda x:295 if x==0 else 300,list(col_9))), label='machine fail')     # will be the class label failuer is 1 and non-failure is 0 (preferaably make it 2) 

# Adding labels and title
plt.xlabel('Index')
plt.ylabel('Values')
plt.title('')
plt.legend()

# Show plot
plt.show()
