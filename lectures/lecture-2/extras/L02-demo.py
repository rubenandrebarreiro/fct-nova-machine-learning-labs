import numpy as np
import matplotlib.pyplot as plt

degree = 2

mat = np.loadtxt('polydata.csv',delimiter=';')
x,y = (mat[:,0], mat[:,1])
coefs = np.polyfit(x,y,degree)

pxs = np.linspace(0,max(x),200)
poly = np.polyval(coefs,pxs)

plt.figure(1, figsize=(12, 8), frameon=False)
plt.plot(x,y,'.r')
plt.plot(pxs,poly,'-')    
plt.axis([0,max(x),-1.5,1.5])    
plt.title(f'Degree: {degree}')
plt.savefig('testplot.png')
plt.close()
