
# coding: utf-8

# In[51]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[53]:


from IPython.core.display import display, HTML
display(HTML("<style>.rendered_html { font-size: 16px; }</style>"))


# ## Locally Weighted Linear Regression
# In  this  problem,  we  will  generalize  the  ideas  of  linear  regression  to  implement  locally  weighted  linear regression where we want to weigh different training examples differently.

# In[19]:


x_data = np.genfromtxt("Assignment_1_datasets/weightedX.csv")
# Normailzation
x_data = (x_data - np.mean(x_data) )/ np.std(x_data) 
x_data = np.vstack((x_data,np.ones_like(x_data)))
x_data = x_data.T
y_data = np.genfromtxt("Assignment_1_datasets/weightedY.csv")


# In[20]:


theta  = np.asarray([0,0]) # theta_1,theta_0
diff = y_data - np.matmul(x_data,theta)
loss = 0.5 * np.matmul(diff.T,diff)


# This is the initial loss with parameters equal to zero.

# In[21]:


loss


# In[22]:


theta_final = np.matmul(np.linalg.inv(np.matmul(x_data.T,x_data)),np.matmul(x_data.T,y_data))


# We obtain the parameters learned by unweighted linear regression implemented using normal equation, the solution being $$\theta = (X^T X)^{-1} X^T Y $$ The first output is the **weight** and second is the **bias**.

# In[40]:


theta_final 


# In[41]:


diff = y_data - np.matmul(x_data,theta_final)
loss = 0.5 * np.matmul(diff.T,diff)


# This is the new loss, it is apparent that the loss has decreased but not by too much.

# In[42]:


loss


# This is the plot of unweighted linear regression output and the data, it is clear that our regression has missed important underlying patterns in the data.

# In[43]:


plt.scatter(x_data[:,0],y_data)
plt.plot(x_data[:,0],theta_final[1] + theta_final[0]*x_data[:,0] ,c="r")


# In[48]:


def loc_weighted_lr(x_data,y_data, input_x, tau = 0.8):
    W = np.diag(np.exp(-np.square(x_data[:,0] - input_x)/(2*tau*tau)))
    theta_loc = np.matmul(np.linalg.inv(np.matmul(np.matmul(x_data.T,W),x_data)),
                          np.matmul(np.matmul(x_data.T,W),y_data))
    return theta_loc[1] + theta_loc[0]*input_x


# After this we implement locally weighted linear regression.  This is not learning per se, as the parameters get learnt at inference time. I did the derivation and found that the normal equation solution for locally weighted LR is $$ \theta = (X^T W X)^{-1} X^T W Y $$ 

# Below is the plot of the output we obtain for locally weighted LR with bandwidth parameter $\tau = 0.8$.

# In[49]:


fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(x_data.T[0],y_data)
outp = []
for i in range(100):
    outp.append(loc_weighted_lr(x_data,y_data,x_data[i,0],tau=0.8))
xs = x_data[:,0]
ys = np.asarray(outp)
xs, ys = zip(*sorted(zip(xs,  ys  )))
ax.plot(xs,ys,c="r")
ax.text(0.7,0.1,"Tau = "+"0.8",fontsize=15)


# Finally we vary the bandwidth $\tau$ and obtain the following plots. It seems that a value of 0.3 for the bandwidth results in the best regression output.

# In[50]:


for t in [0.1,0.3,2,10]:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x_data.T[0],y_data)
    outp = []
    for i in range(100):
        outp.append(loc_weighted_lr(x_data,y_data,x_data[i,0],tau=t))
    xs = x_data[:,0]
    ys = np.asarray(outp)
    xs, ys = zip(*sorted(zip(xs,  ys  )))
    ax.plot(xs,ys,c="r")
    ax.text(0.7,0.1,"Tau = "+str(t),fontsize=15)


# When tau is too small the model starts fitting the small errors in the data and not the underlying pattern, on the other hand when tau is too large the "local" part matters less and we approach unweighted regression, that is the regression starts giving near equal weight to all points no matter how close they are to the target.
