import matplotlib.pyplot as plt
import numpy as np
import time
from mpl_toolkits.mplot3d import Axes3D

wine_x = np.genfromtxt("Assignment_1_datasets/linearX.csv")
wine_x = (wine_x - np.mean(wine_x)) / np.std(wine_x)
wine_x = np.vstack((wine_x,np.ones_like(wine_x)))
wine_x = wine_x.T
wine_y = np.genfromtxt("Assignment_1_datasets/linearY.csv")

theta  = np.asarray([0,0]) # theta_1,theta_0
lr = 0.001

diff = wine_y - np.matmul(wine_x,theta)
loss = 0.5 * np.matmul(diff.T,diff)
ch_loss = 2
itera = 1

def compute_cost(t1,t0):
        diff = wine_y - np.matmul(wine_x,np.asarray([t1,t0]))
        loss = 0.5 * np.matmul(diff.T,diff)
        return loss

theta_0 = np.arange(-2.5, 2.5, 0.01)
theta_1 = np.arange(-2.5, 2.5, 0.01)
theta_1, theta_0 = np.meshgrid(theta_1, theta_0)

cost = [[compute_cost(theta_1[i, j] , theta_0[i,j])
             for j in range(theta_1.shape[0])] for i in range(theta_0.shape[0])]
cost = np.asarray(cost)

fig_1 = plt.figure()
ax_1 = Axes3D(fig_1)
surf_1 = ax_1.plot_surface(theta_0, theta_1, cost)

fig_2 = plt.figure()
ax_2 = fig_2.add_subplot(111)
CS = ax_2.contour(theta_0, theta_1, cost)
ax_2.clabel(CS, inline=1, inline_spacing=0, fontsize=8)

fig_1.show()
fig_2.show()
while np.abs(ch_loss) > 0.00001:
    
    diff = wine_y - np.matmul(wine_x,theta)
    loss = 0.5 * np.matmul(diff.T,diff)
    theta = theta + lr * np.matmul(wine_x.T,diff)
    itera+=1
    diff = wine_y - np.matmul(wine_x,theta)
    loss_new = 0.5 * np.matmul(diff.T,diff) # can be commented out
    ch_loss = loss - loss_new
    time.sleep(0.2)
    ax_1.scatter(theta[1],theta[0],compute_cost(theta[1],theta[0]),c="r")

    ax_2.scatter(theta[1],theta[0],c="r")

    fig_1.canvas.draw()
    fig_2.canvas.draw()
    if itera % 100 == 0:
        print("Iteration - ", itera)
        print("Loss - ",loss_new)
        print("LR - ",lr)
        print("Change in Loss - ",ch_loss)
        print("Theta -- ", theta)
        print("--------------")
