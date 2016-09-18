import numpy as np
import numpy.linalg as la

file = open("data.txt")
data = np.genfromtxt(file, delimiter=",")
file.close()

print "Data matrix\n"
print data

M = np.zeros([len(data[:,0])*2,6])
b = np.zeros([len(data[:,0])*2,1])

for i in range(len(data[:,0])):
	M[(2*i)][0] = data[i,2]
	M[(2*i)][1] = data[i,3]
	M[(2*i)][2] = 1
	M[(2*i)+1][3] = data[i,2]
	M[(2*i)+1][4] = data[i,3]
	M[(2*i)+1][5] = 1
	b[(2*i)][0] = data[i,0]
	b[(2*i)+1][0] = data[i,1]

M = np.matrix(M)
b = np.matrix(b)

print "\nM matrix\n"
print M
print "\nb matrix\n"
print b

a, e, r, s = la.lstsq(M, b)

print "\na\n"
print a
print "\nM*a\n"
print M*a

norm_of_difference = la.norm(M*a-b)
sum_squared_error = norm_of_difference**2

print "\nSum-squared error\n"
print sum_squared_error
print "\nSum-squared error from la.lstsq\n"
print e
