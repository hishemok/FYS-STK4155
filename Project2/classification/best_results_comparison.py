import pandas as pd

column_names = ["Index", "Optimizer", "lambda", "gamma", "learning_rate", "accuracy"]
df = pd.read_csv("Project2/classification/classification_results.csv", skiprows=1,names = column_names) 
df['learning_rate'] = df['learning_rate'].replace('', None)
highest_accuracy_df = df.loc[df.groupby("Optimizer")["accuracy"].idxmax()]
highest_accuracy_df = highest_accuracy_df.reset_index(drop=True)
print(highest_accuracy_df)



column_names = ['Hidden Activation Functions','Hidden Derivatives','Output Activation Function','Cost Function','Cost Derivative','Accuracy (%)'
]
df = pd.read_csv("Project2/classification/Best_functions.csv",skiprows=1,names = column_names)  
df['Accuracy (%)'] = pd.to_numeric(df['Accuracy (%)'], errors='coerce')
best_combination = df.loc[df['Accuracy (%)'].idxmax()]
print("Best combination of functions:")
print(best_combination)

