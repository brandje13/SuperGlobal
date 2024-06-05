import os
import pickle as pkl
import json
from PIL import Image


def process_txt_files(folder_path_txt, folder_path_img):
    data = {'imlist': [], 'qimlist': [], 'gnd': []}
    query_info = {}

    # Iterate through each file in the directory
    for img in os.listdir(folder_path_img):
        # Check if the file ends with .jpg or .png
        # Leave extentions to handle multiple data types
        if img.endswith(".jpg") or img.endswith(".png"):
            # Add the file to the list
            data['imlist'].append(img)#[:-4])

    # Iterate over all text files in the directory
    for filename in os.listdir(folder_path_txt):
        if filename.endswith('.txt'):
            # Determine the category based on the filename
            parts = filename.split('_')
            query_name = '_'.join(parts[:-1])  # Extract query name
            category = parts[-1][:-4]

            # Read the content of the text file
            with open(os.path.join(folder_path_txt, filename), 'r') as file:
                lines = file.readlines()

            # Process each line in the text file
            for line in lines:
                parts = line.split()

                if query_name not in query_info:
                    query_info[query_name] = {'query': None, 'bbx': None,
                                              'ok': [], 'good': [], 'junk': []}

                # Check if the line indicates a query
                if category == 'query':
                    query_info[query_name][category] = parts[0][5:] + '.jpg'
                    #w, h = Image.open(folder_path_img + query_info[query_name][category]).size
                    #query_info[query_name]['bbx'] = [0, 0, w, h]
                    query_info[query_name]['bbx'] = list(map(float, parts[1:]))
                    data['qimlist'].append(parts[0][5:] + '.jpg')

                # Populate data dictionary based on category
                if category in ['ok', 'good', 'junk']:
                    query_info[query_name][category].append(parts[0] + '.jpg')

                    # Add images to 'imlist'
                    # data['imlist'].extend(parts)

    # Populate 'gnd' based on query info
    for query_name, info in query_info.items():
        data['gnd'].append(info)

    return data


# Specify the folder path containing the text files
folder_path_txt = 'revisitop/roxford5k/groundtruth/test'
folder_path_img = 'revisitop/roxford5k/jpg/test'

# Process the text files and store the result in 'result'
result = process_txt_files(folder_path_txt, folder_path_img)

# Save the result as json
with open('./revisitop/roxford5k/gnd_roxford5k.json', 'w') as json_file:
    json.dump(result, json_file, indent=4)

# Save the result as Pickle (pkl)
with open('./revisitop/roxford5k/gnd_roxford5k.pkl', 'wb') as pkl_file:
    pkl.dump(result, pkl_file)
