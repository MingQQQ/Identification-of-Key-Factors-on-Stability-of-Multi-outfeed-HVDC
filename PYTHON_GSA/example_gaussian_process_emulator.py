import gaussian_process_emulator as gpe
import pandas as pd
import time

#Runs the gaussian process emulator example, training and evaluating a model on the included data.
start1=time.clock()
df = pd.read_csv("data.csv")
parameters = ["cbG", "cbL", "dsG", "dsL","lsG", "lsL", "qnG", "qnL","xcG", "xcL"]
quantity_mean = "zhibiao"

testsetsize = 90

gpe.gaussian_process_example(df,parameters,quantity_mean,testsetsize)
end1=time.clock()
print('GPR time',end1-start1)