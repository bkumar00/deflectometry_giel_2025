import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import cv2
from scipy.ndimage import laplace
from scipy.fftpack import dst, idst

'''
lower_base = 668
upper_base = 3025
'''
lower_base = 642
upper_base = 2500

top_limit = 2640

screenwidth = 525
screenheight = 300
distance = 160 #16cm


# Crop the data using the specified limits
datafb = pd.read_csv('filtered_full_brightness2.csv', header=None)
datalr = pd.read_csv('filtered_left_right_gradient2.csv', header=None)
datatb = pd.read_csv('filtered_top_bottom_gradient2.csv', header=None)

datafb = 255 * (datafb - np.min(datafb)) / (np.max(datafb) - np.min(datafb))
datalr = 255 * (datalr - np.min(datalr)) / (np.max(datalr) - np.min(datalr))
datatb = 255 * (datatb - np.min(datatb)) / (np.max(datatb) - np.min(datatb))

full_brightness = datafb.iloc[top_limit:, lower_base:upper_base]
left_right = datalr.iloc[top_limit:, lower_base:upper_base]
bottom_top = datatb.iloc[top_limit:, lower_base:upper_base]




# Plot the first normalized data
plt.figure(figsize=(12, 6))
plt.imshow(full_brightness, cmap='gray')
plt.tight_layout()
plt.show()

'''
Rx = left_right @ np.linalg.inv(full_brightness)
Ry = bottom_top @ np.linalg.inv(full_brightness)
'''
Rx = left_right / full_brightness
Ry = bottom_top / full_brightness


R1 = screenwidth / (screenwidth**2 + distance**2) * (2*Rx - 1)
R2 = screenheight / (screenheight**2 + distance**2) * (2*Ry - 1)
R3 = np.sqrt(1 - R1**2 - R2**2)
v = np.array([-1, 0, 0])

r_vector = np.stack([R1,R2,R3], axis=-1)
print(r_vector.shape)
normal_vector = (r_vector + v)
normal_vector /= np.linalg.norm(normal_vector, keepdims=True)
#array_2d = normal_vector.reshape(-1, 3)

# Save as CSV
#np.savetxt("normal_map.csv", array_2d, delimiter=",", fmt="%.6f")
 
#normal_map_normalized = (normal_vector + 1) / 2
gradient_x = normal_vector[:, :, 0]
gradient_y = normal_vector[:, :, 1]
h, w = gradient_x.shape
height_map = np.zeros((h, w))
for x in range(1, w):
        height_map[0, x] = height_map[0, x-1] + gradient_x[0, x]

    # Integrate first column using G_y
for y in range(1, h):
        height_map[y, 0] = height_map[y-1, 0] + gradient_y[y, 0]
for y in range(1, h):
    for x in range(1, w):
        height_map[y, x] = (height_map[y, x-1] + gradient_x[y, x] + height_map[y-1, x] + gradient_y[y, x]) / 2

plt.imshow(height_map, cmap='gray')
plt.colorbar(label="Height")
plt.title("Reconstructed Height Map")
plt.show()

'''
Printing the normal vector
print(normal_vector.shape)
plt.figure(figsize=(12, 6))
plt.imshow(normal_vector)  # Use viridis for curvature visualization
plt.tight_layout()
plt.show()
'''

normal_map = (normal_vector * 2) - 1
p = normal_map[:, :, 0] / normal_map[:, :, 2]  # z_x = R/Z
q = normal_map[:, :, 1] / normal_map[:, :, 2]  # z_y = G/Z

px = np.gradient(p, axis=1)
qy = np.gradient(q, axis=0)
div = px + qy

def poisson_solver(div):
    m, n = div.shape
    # Perform DST along the rows
    dst_y = dst(div, type=1, axis=0, norm='ortho')
    # Perform DST along the columns
    dst_xy = dst(dst_y, type=1, axis=1, norm='ortho')
    
    # Create the denominator matrix
    denom = (2 * np.cos(np.pi * np.arange(m) / m) - 2)[:, None] + (2 * np.cos(np.pi * np.arange(n) / n) - 2)
    
    # Solve the Poisson equation in the frequency domain
    z = dst_xy / denom
    
    # Perform inverse DST along the columns
    idst_x = idst(z, type=1, axis=1, norm='ortho')
    # Perform inverse DST along the rows
    idst_xy = idst(idst_x, type=1, axis=0, norm='ortho')
    
    return idst_xy

height_map = poisson_solver(div)

# Normalize the height map for visualization
height_map_normalized = (height_map - np.min(height_map)) / (np.max(height_map) - np.min(height_map))

# Save the height map as an image
cv2.imwrite('height_map.png', (height_map_normalized * 255).astype(np.uint8))

print("Height map recovery complete. Saved as 'height_map.png'.")



'''
grad_x = cv2.Sobel(full_brightness, cv2.CV_64F, 1, 0, ksize=5) #applying a sobel filter in x dir and getting gradient
grad_y = cv2.Sobel(full_brightness, cv2.CV_64F, 0, 1, ksize=5) #applying a sobel filter in y dir and getting gradient
magnitude = np.sqrt(grad_x**2 + grad_y**2)
direction = np.arctan2(grad_y, grad_x)
height, width = full_brightness.shape
nx = np.zeros((height, width))
ny = np.zeros((height, width))
nz = np.ones((height, width))  # z component points upwards by default
scale_factor = 1.0  # Adjust based on your sensitivity requirements
nx = -grad_x * scale_factor
ny = -grad_y * scale_factor
norm = np.sqrt(nx**2 + ny**2 + nz**2)
nx /= norm
ny /= norm
nz /= norm
nx_vis = (nx + 1) / 2
ny_vis = (ny + 1) / 2
nz_vis = (nz + 1) / 2
dnx_dx = cv2.Sobel(nx, cv2.CV_64F, 1, 0, ksize=3)
dnx_dy = cv2.Sobel(nx, cv2.CV_64F, 0, 1, ksize=3)
dny_dx = cv2.Sobel(ny, cv2.CV_64F, 1, 0, ksize=3)
dny_dy = cv2.Sobel(ny, cv2.CV_64F, 0, 1, ksize=3)
dnz_dx = cv2.Sobel(nz, cv2.CV_64F, 1, 0, ksize=3)
dnz_dy = cv2.Sobel(nz, cv2.CV_64F, 0, 1, ksize=3)
normal_vis = np.stack([nx_vis, ny_vis, nz_vis], axis=-1)
mean_curvature = np.sqrt(dnx_dx**2 + dny_dy**2 + dnz_dx**2 + dnz_dy**2)
smoothed_curvature = gaussian_filter(mean_curvature, sigma=1.0)
normalized_curvature = cv2.normalize(smoothed_curvature, None, 0, 1, cv2.NORM_MINMAX)

print(normalized_curvature.shape)
plt.figure(figsize=(12, 6))
plt.imshow(normalized_curvature, cmap='viridis')  # Use viridis for curvature visualization
plt.colorbar(label='Normalized Curvature')
plt.title('Surface Curvature Map')
plt.tight_layout()
plt.show()
'''

'''
# Plot the first normalized data
plt.figure(figsize=(12, 6))
plt.imshow(normalized_data1, cmap='gray')
plt.colorbar(label='Normalized Intensity')
plt.title('Normalized Image Data (Method 1)')
plt.tight_layout()
plt.show()
'''