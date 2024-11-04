## Project 2

#### Description:
Regression and neural network analysis. Modelling Franke function and thereafter finding out how to apply these machine learning programs to predict cancer probability, from a dataset: https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data

### Folders:
#### Classification:
Classification.py:
    - Apply Neural Network to cancer data, and finding optimal parameters
classification_regression.py
    - Apply SGD to cancer data, and finding optimal parameters
best_results_comparison.py
    -read the results files to compare who did best in the end
Best_functions.csv
    -Stored all results from combinations of activation functions in the neural network applied to cancer data
Best_functions.txt
    -same as the csv file
Hidden_sizes.txt
    -Find optimal hidden layer sizes for Neural Network on cancer data
Learning_batchsize.txt:
    -Finding optimal learningrate with fixed parameters

#### Data:
dataloader.py
    -Read cancer data and store as pd dataframe
frankefunction.py
    -Creates the Franke function
data.csv
    -Stored cancer data from dataloader.py

#### nn:
neural_network.py
    -The neural Network Class
nn_analysis.py
    -Applying Neural Network to franke function and comparing actiavtion functions
test.py 
    -just a testing file... just ignore it
activation_functions.txt
    -Results from nn_analysis.py

#### Plots:
Saved plots from different analysis

#### Regression:
Regression.py
    -Regression classs. Containing OLS, Ridge, Gradient Descent, Stochastich Gradient Descent, and some more less exciting stuff
regression_analysis.py
    -Testing regression class on the franke function
Learning_rate.txt
    -results stored from experimenting with learning rate
Momentum.txt
    -results stored from experimenting with momentum
mse_array.npy
    -left over file with some values to save time when running regression_analysis
Polynomial_degree.txt
    -finding optimal polynomial degree from regression_analysis.py

