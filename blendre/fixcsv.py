import numpy as np

table = np.loadtxt('matice.csv', delimiter=',')

print(table[:,0].astype(int)-1)
table[:,0] = table[:,0].astype(int)-1
np.savetxt('matice1.csv', table, delimiter=',', fmt='%d,%f,%f,%f,%f,%f,%f,%f,%f,%f')
           
