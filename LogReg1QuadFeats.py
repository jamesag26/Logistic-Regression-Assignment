#Author: James Alford-Golojuch
#Used tutorial for logistic regression with gradient descent for help with basis of code
#http://aimotion.blogspot.com/2011/11/machine-learning-with-python-logistic.html
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from scipy.special import expit

#For plotting logistic regression function
# define our hypothesis (vectorized!)
def f(x,thetas): 
  return expit(np.matrix(thetas)*x);

#For plotting logistic regression function
# basis expansion
expand = lambda x,y: np.vstack((
    np.ones(x.size), # add the bias term
    x.ravel(), # make the matrix into a vector
    y.ravel(),
    x.ravel() * y.ravel(),
    x.ravel()**2,
    y.ravel()**2))

#Sigmoid function implementation
def sigmoid(z):
    return 1/(1 + (np.exp(-1 * z)))

#Cost function
def costfunc(theta,data,labels):
    #Finds predicted probability of label 1
    probOf1 = sigmoid(np.dot(data,theta))
    #Finds log likelihood vector
    logOf1 = (-1 * labels) * np.log(probOf1) - (1 - labels) * np.log(1 - probOf1)
    return logOf1.mean()
    
#gradient descent
def gradDec(theta,data,labels):
    #Finds predicted probability of label 1
    probOf1 = sigmoid(np.dot(data, theta))
    #Difference between label and prediction
    error = probOf1 - labels
    #Finds gradient vector
    grad = np.dot(error, finalDataQuad) / labels.size
    return grad
    
def predict(theta, data, labels):
    correct = 0
    total = 0
    probOf1 = sigmoid(np.dot(data, theta))
    for x in range(0,len(probOf1)):
        if ((probOf1[x] > .5 and labels[x] == 1) or (probOf1[x] <= .5 and labels[x] == 0)) :
            correct = correct + 1
        total = total + 1
    return correct / total
    
#Test and train data contains 2 columns 
trainData1 = np.loadtxt('synthetic-6.csv', dtype='float', delimiter=',')

data = trainData1[:,0:2]
labels = trainData1[:,2]

thetaInit = 0.1* np.random.randn(6)
#Basis expansion to qaudratic features
#Append column of 1s
dataAdd1 = np.append( np.ones((data.shape[0], 1)), data, axis=1)

dataQuad = [(data[0,0] * data[0,1]),(data[0,0] * data[0,0]),(data[0,1] * data[0,1])]
for x in range(1,len(labels)):
    temp = [(data[x,0] * data[x,1]),(data[x,0] * data[x,0]),(data[x,1] * data[x,1])]
    dataQuad = np.vstack((dataQuad,temp))
finalDataQuad = np.append( dataAdd1,dataQuad, axis=1)

print(finalDataQuad)
#Find final theta values for logistic regression function
thetaFinal = opt.fmin_bfgs(costfunc, thetaInit, fprime=gradDec, args=(finalDataQuad, labels))


# create the domain for the plot
x_min =min(data[:,0]) - 1; x_max = max(data[:,0]) + 1
y_min =min(data[:,1]) - 1; y_max = max(data[:,1]) + 1
x1 = np.linspace(x_min, x_max, 200)
y1 = np.linspace(y_min, y_max , 200)
x,y = np.meshgrid(x1, y1)

# evalute it in a vectorized way (and reshape into a matrix)
data = expand(x,y) 
z = f(data,thetaFinal)
z = z.reshape(x.shape)

#Prints scatter plot of data
plt.figure(1)
plt.title("Gradient Descent w/ Quadratic Features: Synthetic-6")
plt.scatter(trainData1[:,0], trainData1[:,1], c=trainData1[:,2], alpha=0.5)

# show the function value in the background
plt.set_cmap('prism')
cs = plt.imshow(z,
  extent=(x_min,x_max,y_max,y_min), # define limits of grid, note reversed y axis
  cmap=plt.cm.jet)
plt.clim(0,1) # defines the value to assign the min/max color

# draw the line on top
levels = np.array([.5])
cs_line = plt.contour(x,y,z,levels)

# add a color bar
CB = plt.colorbar(cs)
plt.show()

#Misclassification rate
p = predict(thetaFinal,finalDataQuad,labels)
print("The misclassification rate is ", 1-p, " %")