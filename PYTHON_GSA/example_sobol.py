import pandas as pd
import matplotlib.pyplot as plt
from sensitivity_analysis_sobol import sobol_indicies_from_emulator
import time
start2=time.clock()
df = pd.read_csv("data.csv")

parameters = ["cbG", "cbL", "dsG", "dsL","lsG", "lsL", "qnG", "qnL","xcG", "xcL"]
bounds = [ [0, 4840], [218.5, 5058.5],[4765, 10010], [0, 5245], [0, 3023.8],[569.8,3593.6], [366.3, 3260],[0, 2893.7], [4290.1, 7270], [0, 2979.9]]
quantity_mean = "zhibiao"  # Name of the column containing the mean of the output quantity.

N = 2 ** 12 # Note: Must be a power of 2.  (N*(2D+2) parameter value samples are used in the Sobol analysis.

Si_df = sobol_indicies_from_emulator(df, parameters, bounds, quantity_mean, N)  # Perform analysis
print("The first and total order Sobol sensitivity indicies, and their 95% confidence intervals, are: ")
print(Si_df)
end2 = time.clock()
print('Sobol time', end2 - start2)

labels = Si_df.index.values
S1 = Si_df ["S1"]
S_interaction = Si_df["ST"]-Si_df["S1"]
width = 0.95
fig, ax = plt.subplots(figsize=(10.5, 6))
ax.bar(labels, S1, width, label='Main effect',color='lightskyblue',hatch='/', edgecolor = "black")#hatch='red'
ax.bar(labels, S_interaction, width, bottom=S1, label='Interaction', color='cornflowerblue', edgecolor="black")#hatch='/'
ax.set_xlabel('Parameters', fontsize=15)
x=range(0,10,1)
plt.xticks(x,("cbG", "cbL", "dsG", "dsL","lsG", "lsL", "qnG", "qnL","xcG", "xcL"),fontsize=10)#
ax.set_title('Sobol Sensitivity of Model Parameters', fontsize=15)
ax.set_ylim([0, 0.5])
ax.legend(fontsize=15)
plt.show()








