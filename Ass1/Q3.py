
# coding: utf-8

# ## Logistic Regression using Newton's Method
# In this part we implement classification using logistic regression using Newton's method for optimization.

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


x_data = np.genfromtxt("Assignment_1_datasets/logisticX.csv",delimiter=',')
x_data = (x_data - np.mean(x_data,axis=0) ) / np.std(x_data,axis=0)
x_data = np.hstack((x_data,np.ones((100,1))))
y_data = np.genfromtxt("Assignment_1_datasets/logisticY.csv")


# In[4]:


def sigmoid(x):
    return 1/(1+np.exp(-x))


# In[5]:


theta  = np.asarray([0,0,0]) # theta_1,theta_0
lr = 1
iteration = 1


# We calculate the Grad and Hessian matrices and use the following update rule to update our parameters (initialized at zero).
# $$ \theta_{t+1} = \theta_{t} - H(f(\theta_t))^{-1} \nabla f(\theta_t) $$ 

# Following is the output of our optimization. The first parameter corresponds to the first feature, second to the second feature and last one is the bias. It is clear that our algorithm converges sufficiently in only one iteration and after this all the parameters are just scaled by some number. After five iterations overflow is encountered and the algorithm terminates.

# In[6]:


while True:
    hx = sigmoid(np.matmul(x_data,theta))
    grad = np.matmul(x_data.T,y_data - hx)
    hessian = np.asarray([[np.sum(x_data[:,j]*x_data[:,k]*(1-hx)*hx) for j in range(3)] for k in range(3)])
    try:
        
        theta = theta - lr*np.matmul(np.linalg.inv(hessian),grad)
        print ("Iteration = ", iteration)
        print ("Theta array = " ,theta)
    except:
        print("Overflow in iteration",iteration)
        break
    iteration+=1


# Thus a reasonable equation of the boundary is $ -8.97341408e-01 * x_0 + 9.20112283e-01* x_1 + 3.85905135e-16

# In[8]:


theta = [ -8.97341408e-01  , 9.20112283e-01   ,3.85905135e-16]


# Below is the plot of our decision boundary and the data points, it can be seen that our boundary separates the two classes reasonably well.

# In[9]:


fig = plt.figure()
ax = fig.add_subplot(111)
markers = ["s","o"]
colors = ["b","r"]
for i, c in enumerate(np.unique(y_data)):
    ax.scatter(x_data[:,0][y_data==c],x_data[:,1][y_data==c],c=colors[i], marker=markers[i])
ax.plot(x_data[:,0],(-theta[2]-theta[0]*x_data[:,0] )/ theta[1])

