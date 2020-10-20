# -*- coding: utf-8 -*-
"""
Created on Fri May 15 16:27:51 2020

@author: rishbans
"""

#Initialize Learning rate, bias and weights
lr = 1
bias = 1
weights = [-50, 20, 20]

#perceptron(x_1, x_2, output) 
def perceptron(x_1, x_2, output):
    outputP = bias*weights[0] + x_1*weights[1] + x_2*weights[2]
    if outputP > 4.6:
        outputP = 1
    else:
        outputP = 0
    error = 1/2*(outputP - output)**2
    weights[0] = weights[0] + error*bias*lr
    weights[1] = weights[1] + error*x_1*lr
    weights[2] = weights[2] + error*x_2*lr

#predict(x_1, x_2)
def predict(x_1, x_2):
    outputP = bias*weights[0] + x_1*weights[1] + x_2*weights[2]
    if outputP > 4.6:
        outputP = 1
    else:
        outputP = 0
    return outputP


#Call perceptron for each row of OR gate
#Run in Loop for multiple times to train the Network
for i in range(50):
    perceptron(0,0,0)
    perceptron(0,1,1)
    perceptron(1,0,1)
    perceptron(1,1,1)    

#Take Input values from user to predict the value 
print("Enter first input")
x_1 = int(input())
print("Enter second input")
x_2 = int(input())
output_predict = predict(x_1,x_2)
print(x_1, "or", x_2, "is:  ", output_predict)









