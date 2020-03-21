import os
from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np
import os
plt.rcParams['figure.figsize'] = [16, 8]


A = imread('images/dog.jpg')
A = np.mean(A, -1); # Convert RGB to grayscale


img = plt.imshow(A)
img.set_cmap('gray')
plt.axis('off')
plt.show()
# np.savetxt("test.csv", A[1:10], delimiter=",")
# plt.savefig('images/grey_dog.png')
print(A.shape)


# Perform full matrix SVD using numpy.linalg.svd 
U, S, VT = np.linalg.svd(A,full_matrices=False)
S = np.diag(S)


# Create the X using all ranks of U, S, VT.
# Use library os and the function getsize to determine the total number of kilo bytes of # the full image.
X = U*S*np.transpose(VT)
Xsize = os.path.getsize(X)
Xsize


j = 0
for r in (5, 20, 100):
    # Construct approximate image
    Xapprox = U[:,:r] @ S[0:r,:r] @ VT[:r,:]
    plt.figure(j+1)
    j += 1
    img = plt.imshow(Xapprox)
    img.set_cmap('gray')
    plt.axis('off')
    plt.title('r = ' + str(r))
    plt.show()