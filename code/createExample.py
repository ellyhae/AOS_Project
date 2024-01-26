import matplotlib.pyplot as plt
import os
import cv2

image_name = '0_7_6_-2_integral'  # without file ending
results_path = 'results'
dataset_path = 'test'
save_path = 'examples'

os.makedirs(save_path, exist_ok=True)

integral_path = os.path.join(dataset_path, image_name + '.tiff')
gt_path = os.path.join(dataset_path, image_name.removesuffix('integral') + 'gt.png')
output_path = os.path.join(results_path, image_name + '.png')

integral = cv2.imreadmulti(integral_path)[1][0]
ground_truth = cv2.imread(gt_path)[..., 0]
output = cv2.imread(output_path)[..., 0]


fig, [inp, out, gt] = plt.subplots(1, 3, sharey=True, figsize=(6, 2.5))

inp.imshow(integral, vmin=0, vmax=255, cmap='gray')
out.imshow(output, vmin=0, vmax=255, cmap='gray')
gt.imshow(ground_truth, vmin=0, vmax=255, cmap='gray')

inp.set_title('Input')
out.set_title('Output')
gt.set_title('Ground Truth')
plt.tight_layout(pad=0)
plt.savefig(os.path.join(save_path, image_name + '.png'), transparent=True, dpi=500)