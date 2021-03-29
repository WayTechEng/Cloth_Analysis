import cv2
import numpy as np
import sys
import time
import timeit
import signal
import math
import shutil
import os
import csv


curdir = os.chdir("../")
curdir = os.getcwd()
target_combined_image_location = curdir + "\\All_images"
goal_prefix = curdir + "\\Goals\\"
source = curdir + "\\User_Study_Trials_18_participants"
csv_input = curdir + "\\Analysis\\Analysis.csv"
data = []  # 0,1,2,3 =  2D_No_Sim, AR_No_Sim, 2D_Sim, AR_Sim
configurations = [goal_prefix + "C1.jpg", goal_prefix + "C2.jpg", goal_prefix + "C3.jpg", goal_prefix + "C4.jpg"]
masked_configurations = []
output_data = []  # Configuration, percentage accuracy
print("Source is: ", source)
print("Target dir is :", target_combined_image_location)

lower_blue = np.array([0, 0, 0])
upper_blue = np.array([255, 255, 110])
c_x = 90
c_y = 20
c_h = 420
c_w = 520
#
# c2_x = 70
# c2_y = 40
# c2_h = 420
# c2_w = 520

c2_x = 50
c2_y = 20
c2_h = 450
c2_w = 550

def mask_goal_states(masked_configurations):
    for c in configurations:
        img = cv2.imread(c)
        hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv_img, lower_blue, upper_blue)
        mask = mask[c_y:c_y+c_h, c_x:c_x+c_w]

        # Pad these images out

        # cv2.imshow("masked image", mask)
        # cv2.waitKey(0)
        masked_configurations.append(mask)

    return masked_configurations


def rename_raw_images():
    print("Renaming images in: ")
    print(target_combined_image_location)

    files = os.listdir(target_combined_image_location)
    print(files)
    for file in files:
        print(file)
        new_name = ""
        counter = 0
        for c in file:
            if c == "_":
                counter += 1
            if counter == 4:
                break
            new_name = new_name + c
        print("New name is: ", new_name)

def move_images_to_one_directory(source_dir, target_dir):
    print("Working directory is: ", source_dir)
    print("Moving images to the target directory: ", target_dir)
    sub_dirs = os.listdir(source_dir)
    # print(sub_dirs)
    count = 0
    image_dir = source_dir + "\\"
    for d in sub_dirs:
        dir = image_dir + d + "\\images"
        files = os.listdir(dir)
        # print(files)
        for file in files:
            directory = dir + "\\" + file
            shutil.move(directory, target_dir)
            print(file)
            count += 1

    print("Finished moving the files")
    print("Files moved: ", count)

# 2D_No_Sim, AR_No_Sim, 2D_Sim, AR_Sim
def determine_mode(name):
    found_first__ = False
    counter = 0
    mode = -1
    p = -1
    p_str = ""
    collapsed_name = ""
    for c in name:
        if c == "_":
            counter += 1
            found_first__ = True
            if counter == 4:
                break
        if found_first__:
            if c != "_":
                collapsed_name = collapsed_name + c
        else:
            p_str += c
    p = int(p_str)

    if "2d" in collapsed_name.lower():
        if "nosim" in collapsed_name.lower():
            mode = 0
        elif "sim" in collapsed_name.lower():
            mode = 2
    elif "ar" in collapsed_name.lower():
        if "nosim" in collapsed_name.lower():
            mode = 1
        elif "sim" in collapsed_name.lower():
            mode = 3

    if (mode == -1) or (p == -1):
        print("Program failed")
        sys.exit(0)

    return p, mode, collapsed_name


def read_csv(d):
    print("reading csv file")
    start = 9
    stop = 25
    counter = 0
    p_mode_config = []
    with open(csv_input, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            counter += 1
            if counter >= start:
                id = row[0]
                temp_arr= []
                for i in range(0,4):
                    temp_arr.append([row[i*2 + 1], row[i*2 + 2]])
                p_mode_config.append([id, temp_arr])
                # print(row)
            if counter == stop:
                break
            # print(counter)

    return p_mode_config


def find_correct_image(id, mode, cvs_data):
    counter = 0
    for d in cvs_data:
        if int(d[0]) == id:
            # print("mode/config: ", d[1])
            # print(mode)
            found_mode = False
            for m in d[1]:
                # print(int(m[0]))
                if int(m[0]) == mode:
                    found_mode = True
                    return int(m[1])
            if not found_mode:
                print("Failed to match the modes")
            break


def process_image(name, config, mode, output_data):
    while True:
        print("Processing image against configuration: ", config)
        path = target_combined_image_location + "\\" + name
        img = cv2.imread(path)
        desired_img = masked_configurations[config - 1]
        hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv_img, lower_blue, upper_blue)
        mask = mask[c2_y:c2_y + c2_h, c2_x:c2_x + c2_w]

        ## Overlay the image
        overlaid_image = np.copy(mask)
        print(mask.shape[0])
        print(mask.shape[1])
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                p_mask = mask[i,j]
                p_des = desired_img[i,j]

                if p_mask == p_des:
                    if p_mask == 255:
                        overlaid_image[i,j] = 175
                elif p_des == 255:
                    overlaid_image[i,j] = 75

        ## calculate difference in number of pixels
        pixel_desired = np.sum(masked_configurations[config-1]) / 255
        pixel_actual = np.sum(mask) / 255
        diff = abs(pixel_actual - pixel_desired)
        perc = diff/(pixel_actual) * 100
        # print(perc)
        output_data.append([config, mode, perc])

        ## Show image
        cv2.imshow("Achieved state", mask)
        cv2.imshow("Desired state", masked_configurations[config-1])
        cv2.imshow("Overlaid image", overlaid_image)
        k = cv2.waitKey(10)
        if k == ord('n'):
            break
        elif k == a:


    return output_data



## Code here
masked_configurations = mask_goal_states(masked_configurations)

csv_data = None
csv_data = read_csv(csv_data)
print(csv_data)

# move_images_to_one_directory(source, target_combined_image_location)
# rename_raw_images()

image_names = os.listdir(target_combined_image_location)
# print(image_names)
# file_path = target_combined_image_location + "\\" + image_names[0]
# im = cv2.imread(file_path)

# mode, name = determine_mode(image_names[0])
# print(mode, "  ", name)

for n in image_names:
    participant, mode, name = determine_mode(n)
    data.append([participant, mode])
    # print(participant, "   ", mode, "  ", name)

print(data)
counter = 0
for d in data:
    # print(d)
    p = d[0]
    mode = d[1]
    config = find_correct_image(p, mode, csv_data)
    print(config)
    if config != 1:
        output_data = process_image(image_names[counter], config, mode, output_data)

    counter += 1
print(counter)
# print(output_data)
arr = np.array(output_data)
print(arr)

with open(curdir + "\\Analysis\\Analysis_Output_test.csv", 'w') as f:
    writer = csv.writer(f)

    # for row in arr:
    #     writer.writerow(row)
    writer.writerows(arr)


# cv2.imshow("Raw Image", im)
# cv2.waitKey(0)
print("Program Finished Correctly")
sys.exit(0)






