#Author: James Alford-Golojuch
#Used tutorial for logistic regression with gradient descent for help with basis of code
#http://aimotion.blogspot.com/2011/11/machine-learning-with-python-logistic.html
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

#For plotting logistic regression function
# basis expansion
expand = lambda x,y: np.vstack((
   np.ones(x.size), # add the bias term
   x.ravel(), # make the matrix into a vector
   y.ravel()))
    
#Test and train data contains 2 columns 
trainData1 = np.loadtxt('synthetic-6.csv', dtype='float', delimiter=',')

data = trainData1[:,0:2]
labels = trainData1[:,2]

thetaInit = 0.1* np.random.randn(3)
#Append column of 1s
dataAdd1 = np.append( np.ones((data.shape[0], 1)), data, axis=1)
#Find final theta values for logistic regression function
#thetaFinal = opt.fmin_bfgs(costfunc, thetaInit, fprime=gradDec, args=(dataAdd1, labels))
logistic = LogisticRegression()
logistic.fit(dataAdd1,labels)
y_pred = logistic.predict(dataAdd1)
class_rate = np.mean(y_pred.ravel() == labels.ravel())

# create the domain for the plot
x_min =min(data[:,0]) - 1; x_max = max(data[:,0]) + 1
y_min =min(data[:,1]) - 1; y_max = max(data[:,1]) + 1
x1 = np.linspace(x_min, x_max, 200)
y1 = np.linspace(y_min, y_max , 200)
x,y = np.meshgrid(x1, y1)

# evalute it in a vectorized way (and reshape into a matrix)
data2 = expand(x,y).T 
#Returns probability of each class
z = logistic.predict_proba(data2)
#take first set of probabilities which is probability of class 1
p_1 = z[:,0]
p_1 = p_1.reshape(x.shape)
#Prints scatter plot of data
plt.figure(1)
plt.title("High Level Log Reg w/ Raw Features: Synthetic-6")
plt.scatter(trainData1[:,0], trainData1[:,1], c=trainData1[:,2], alpha=0.5)

# show the function value in the background
plt.set_cmap('prism')
cs = plt.imshow(p_1,
  extent=(x_min,x_max,y_max,y_min), # define limits of grid, note reversed y axis
  cmap=plt.cm.jet)
plt.clim(0,1) # defines the value to assign the min/max color

# draw the line on top
levels = np.array([.5])
cs_line = plt.contour(x,y,p_1,levels)

# add a color bar
CB = plt.colorbar(cs)
plt.show()

print("The misclassification rate is ",1-class_rate)