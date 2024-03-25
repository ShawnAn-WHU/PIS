import os
import numpy as np
from PIL import Image

'''
image_dir = '/home/anxiao/Datasets/Potsdam/2_Ortho_RGB'
label_dir = '/home/anxiao/Datasets/Potsdam/5_Labels_all_noBoundary'
image_output_dir = '/home/anxiao/Datasets/Potsdam/2_Ortho_RGB_256'
label_output_dir = '/home/anxiao/Datasets/Potsdam/5_Labels_all_noBoundary_256'
image_size = (256, 256)

image_files = [i for i in os.listdir(image_dir) if i.endswith('.tif')]
label_files = [l for l in os.listdir(label_dir) if l.endswith('.tif')]
'''
'''
image_dir = '/home/anxiao/Datasets/LoveDA/Combine/Images/Urban'
label_dir = '/home/anxiao/Datasets/LoveDA/Combine/Labels/Urban'
image_output_dir = '/home/anxiao/Datasets/LoveDA/Combine_256/Images/Urban'
label_output_dir = '/home/anxiao/Datasets/LoveDA/Combine_256/Labels/Urban'
image_size = (256, 256)

image_files = [i for i in os.listdir(image_dir) if i.endswith('.png')]
label_files = [l for l in os.listdir(label_dir) if l.endswith('.png')]
'''

image_A_dir = '/home/anxiao/Datasets/LEVIR/train/A'
image_B_dir = '/home/anxiao/Datasets/LEVIR/train/B'
label_dir = '/home/anxiao/Datasets/LEVIR/train/label'
image_A_output_dir = '/home/anxiao/Datasets/LEVIR/train_256/A'
image_B_output_dir = '/home/anxiao/Datasets/LEVIR/train_256/B'
label_output_dir = '/home/anxiao/Datasets/LEVIR/train_256/label'
image_size = (256, 256)

image_A_files = [i for i in os.listdir(image_A_dir) if i.endswith('.png')]
image_B_files = [i for i in os.listdir(image_B_dir) if i.endswith('.png')]
label_files = [l for l in os.listdir(label_dir) if l.endswith('.png')]

for image_file in image_A_files:
    image = Image.open(os.path.join(image_A_dir, image_file))

    width, height = image.size

    num_cols = width // image_size[0]
    num_rows = height // image_size[1]

    for row in range(num_rows):
        for col in range(num_cols):
            left = col * image_size[0]
            upper = row * image_size[1]
            right = left + image_size[0]
            lower = upper + image_size[1]

            small_image = image.crop((left, upper, right, lower))
            small_image.save(os.path.join(image_A_output_dir, f'{image_file[:-4]}_{row}_{col}.png'))
    
print('image A done')

for image_file in image_B_files:
    image = Image.open(os.path.join(image_B_dir, image_file))

    width, height = image.size

    num_cols = width // image_size[0]
    num_rows = height // image_size[1]

    for row in range(num_rows):
        for col in range(num_cols):
            left = col * image_size[0]
            upper = row * image_size[1]
            right = left + image_size[0]
            lower = upper + image_size[1]

            small_image = image.crop((left, upper, right, lower))
            small_image.save(os.path.join(image_B_output_dir, f'{image_file[:-4]}_{row}_{col}.png'))
    
print('image B done')

for label_file in label_files:
    label = Image.open(os.path.join(label_dir, label_file))

    width, height = image.size

    num_cols = width // image_size[0]
    num_rows = height // image_size[1]

    for row in range(num_rows):
        for col in range(num_cols):
            left = col * image_size[0]
            upper = row * image_size[1]
            right = left + image_size[0]
            lower = upper + image_size[1]

            small_label = label.crop((left, upper, right, lower))
            small_label.save(os.path.join(label_output_dir, f'{label_file[:-4]}_{row}_{col}.png'))
    
print('label done')
