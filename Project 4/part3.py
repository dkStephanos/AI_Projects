import pandas as pd
from NeuralNetwork import NeuralNetwork

white_df = pd.read_csv('winequality-white.csv')
NeuralNetwork.classify(white_df, .05, (512,256,128,64,32,16), 'relu', 'adam', .2, 'quality')

red_df = pd.read_csv('winequality-red.csv')
NeuralNetwork.classify(red_df, .05, (512,256,128,64,32,16), 'relu', 'adam', .2, 'quality')