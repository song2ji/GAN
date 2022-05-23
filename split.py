import cv2
import os
import numpy as np
#from matplotlib import pyplot as plt
from glob import glob
from random import randint

data = glob('/home/psdz/pr/mulgan-cvpr/output/128_shortcut1_inject1_none/sample_testing/*.png')
#128_shortcut1_inject1_none/sample_testing/
path = '/home/psdz/pr/mulgan-cvpr/output/128_shortcut1_inject1_none/sample_testing/niter_000_003.png'
cimg = cv2.imread(path,1)
print(cimg.shape)

for k in range(12):
	for i in range(16):
		img = cimg[i*128+(1+i)*2:(i+1)*128+(1+i)*2,k*128+(1+k)*2:(k+1)*128+(1+k)*2,:]
		cv2.imwrite('/home/psdz/pr/mulgan-cvpr/result1/'+str(k)+ str(i)+'.jpg',img )

