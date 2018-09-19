python main.py perceptron iris.data
python main.py adaline iris.data
python main.py sgd iris.data
python main.py multiclass iris.data

python main.py perceptron winequality-red.csv
python main.py adaline winequality-red.csv
python main.py sgd winequality-red.csv
python main.py multiclass winequality-red.csv


- Note that the data set given to the program should not contain any headers.
- You can run test_params.py with different classifiers and data sets to plot the errors/costs for different values of eta.