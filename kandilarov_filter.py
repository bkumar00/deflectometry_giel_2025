import numpy as np
import pandas as pd

file_path = "top_bottom_gradient1.csv"

'''
lower_base = 668
upper_base = 3025
[:, lower_base:upper_base]
'''
rawImage = np.loadtxt(file_path, delimiter=',')
num_columns = rawImage.shape[1]
num_rows = rawImage.shape[0]
filteredImage = np.zeros_like(rawImage)  # Initialize filteredImage with the same shape as rawImage
#read csv 1D - 0%
zeroBrightness = np.loadtxt('0_csv.csv', delimiter=',')
#read csv 1D - 70%
seventyBrightness = np.loadtxt('70_csv.csv', delimiter=',')
#mean_br = np.mean(70% - 0%)
#meanBrightness = np.sum(seventyBrightness - zeroBrightness) / len(seventyBrightness)
meanBrightness = np.mean(seventyBrightness - zeroBrightness)
# p_corr(x) = ( mean_br(x)/(70(x) - 0(x)) ) * (Raw(x) - D(x))
for i in range(num_rows):
    for j in range(num_columns):
        filteredImage[i,j] = (meanBrightness/(seventyBrightness[j] - zeroBrightness[j])) * (rawImage[i,j] - zeroBrightness[j])
#save the filtered image
output_file_name = f"filtered_{file_path.split('.')[0]}.csv"
np.savetxt(output_file_name, filteredImage, delimiter=',')