import numpy as np


def power_iteration(A, epsilon = 0.01, max_iter = 200):
	"""function using power interation method 
	to calculate largest eigenvalue and associated eigenvector"""
	
	N = len(A)
	x = np.random.randn(N)
	mu = ((x.reshape([1,-1])).dot(A).dot(x.reshape([-1, 1]))[0]/(x.reshape([1,-1])).dot(x.reshape([-1,1]))[0])[0]

	for k in range(max_iter):
		x  = A.dot(x.reshape([-1, 1]))
		temp = ((x.reshape([1,-1])).dot(A).dot(x.reshape([-1, 1]))[0]/(x.reshape([1,-1])).dot(x.reshape([-1,1]))[0])[0]
		if abs(temp-mu)<epsilon:
			mu = temp
			break
		mu = temp
		print(k, mu)

	return mu, x/np.sqrt(x.reshape([1,-1]).dot(x.reshape([-1, 1])))


if __name__ == "__main__":


	A = np.array([[7./2, 1./4, 1./4, 1./4, 1./7], [1./5, 8./3, 0., 1./6, 1./8],\
		 [1./2, 1./4, 10., 1./9, 1./3],[1./6, 1./2, 2., 9./7, 1./5],\
		 [1./3, 0., 1., 1./2, 11./2]])
	
	# 2.	
	lambda_1, x = power_iteration(A)
	print("largest eigenvalue: {}, eigenvector: {}".format(lambda_1, x))


	# 3.
	lambda_1, y = power_iteration(A.T)
	print("A transpose, largest eigenvalue: {}, eigenvector: {}".format(lambda_1, y))


	# 4. 
	B = A - lambda_1*(x.reshape([-1,1]).dot(x.reshape([1, -1])))/(x.reshape([1, -1]).dot(x.reshape([-1, 1])))[0]

	print("B: {}".format(B))


	# 5. 

	lambda_2, _ = power_iteration(B)

	print("second largest: {}".format(lambda_2))


	# 6. 

	print("final result: {}".format(lambda_1*lambda_2))










