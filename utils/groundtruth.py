import os
import shutil
from PIL import Image
import json


def create_groundtruth(query_paths, dir_path, dataset):
    # Added qtxtlist and gnd_txt to cleanly separate the modalities
    data = {'imlist': [], 'qimlist': [], 'gnd': [], 'qtxtlist': [], 'gnd_txt': [], 'path': str}
    query_info = {}
    data['path'] = os.path.join(dir_path, dataset)

    # Determine the category based on the filename
    if os.name == 'nt':
        split = "\\"
    elif os.name == 'posix':
        split = "/"
    else:
        split = "_"  # TODO: Better fix for this

    # Iterate through each file in the directory
    for img in sorted(os.listdir(os.path.join(dir_path, dataset))):
        # Check if the file ends with .jpg, .jpeg or .png
        if img.endswith((".jpg", ".png", ".jpeg")):
            # Add the file to the list
            data['imlist'].append(img)

    if all(os.path.isdir(path) for path in query_paths) and (dataset == "ILIAS" or dataset == "ILIAS_Test"):
        for path in query_paths:
            temp_queries = []
            class_pos_images = []
            temp_path_query = os.path.join(path, "query")
            temp_path_pos = os.path.join(path, "pos")

            # A. Gather all positive images for this specific class first
            for file in sorted(os.listdir(temp_path_pos)):
                if file.endswith((".jpg", ".png", ".jpeg")):
                    pos_file = os.path.join("queries", path.split('\\')[-1], 'pos', file)
                    data['imlist'].append(pos_file)
                    class_pos_images.append(pos_file)

            # B. Process Image Queries and Text Queries
            for file in sorted(os.listdir(temp_path_query)):
                # --- IMAGE QUERY LOGIC ---
                if file.endswith((".jpg", ".png", ".jpeg")):
                    query_file = os.path.join(path.split('\\')[-1], 'query', file)
                    query_name = file.split('.')[0]
                    temp_queries.append(query_file)

                    if query_name not in query_info:
                        bbx_path = os.path.join(temp_path_query, query_name + '_bbox.txt')
                        with open(bbx_path, 'r') as f:
                            content = f.read().strip()
                            raw_bbx = list(map(float, content.split()))
                            x, y, w, h = raw_bbx
                            x2 = x + w
                            y2 = y + h
                            bbx = [x, y, x2, y2]

                        query_info[query_file] = {
                            'query': query_file,
                            'bbx': bbx,
                            'ok': class_pos_images.copy(),
                            'good': [],
                            'junk': []
                        }
                        data['qimlist'].append(query_file)

                # --- TEXT QUERY LOGIC ---
                elif file.endswith(".txt") and file.startswith("T"):
                    text_query_id = file.split('.')[0]  # e.g., 'T000'

                    with open(os.path.join(temp_path_query, file), 'r', encoding='utf-8') as f:
                        text_string = f.read().strip()

                    data['qtxtlist'].append(text_query_id)
                    data['gnd_txt'].append({
                        'query': text_query_id,
                        'text': text_string,
                        'ok': class_pos_images.copy(),
                        'good': [],
                        'junk': []
                    })
    else:
        # Iterate over all image files in the directory
        for filename in sorted(query_paths):
            query_name = filename.split(split)[-1]  # Extract query name
            category = 'query'

            # Process each line in the text file
            if query_name not in query_info:
                query_info[query_name] = {'query': None, 'bbx': None,
                                          'ok': [], 'good': [], 'junk': []}

            # Check if the line indicates a query
            if category == 'query':
                query_info[query_name][category] = query_name
                w, h = Image.open(filename).size
                query_info[query_name]['bbx'] = [0, 0, w, h]
                data['qimlist'].append(query_name)

            # Populate data dictionary based on category
            if category in ['ok', 'good', 'junk']:
                query_info[query_name][category].append(None)

    # Populate 'gnd' based on image query info
    for query_name, info in query_info.items():
        data['gnd'].append(info)

    # Save the result as json
    with open(os.path.join(dir_path, dataset, f'gnd_{dataset}.json'), 'w') as json_file:
        json.dump(data, json_file, indent=4)


# Create groundtruth based on provided txt files
def create_groundtruth_from_txt(dir_path, dataset):
    data = {'imlist': [], 'qimlist': [], 'gnd': [], 'path': str}
    query_info = {}
    data['path'] = os.path.join(dir_path, dataset)

    # Iterate through each file in the directory
    for img in sorted(os.listdir(os.path.join(dir_path, dataset))):
        # Check if the file ends with .jpg or .png
        # Leave extentions to handle multiple data types
        if img.endswith(".jpg") or img.endswith(".png") or img.endswith(".jpeg"):
            # Add the file to the list
            data['imlist'].append(img)

    # Iterate over all text files in the directory
    for filename in sorted(os.listdir(os.path.join(dir_path, dataset, "groundtruth"))):
        if filename.endswith('.txt'):
            # Determine the category based on the filename
            parts = filename.split('_')
            query_name = '_'.join(parts[:-1])  # Extract query name
            category = parts[-1][:-4]

            # Read the content of the text file
            with open(os.path.join(os.path.join(dir_path, dataset, "groundtruth"), filename), 'r') as file:
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
                    query_info[query_name]['bbx'] = list(map(float, parts[1:]))
                    data['qimlist'].append(parts[0][5:] + '.jpg')
                    shutil.copy(os.path.join(dir_path, dataset, parts[0][5:] + '.jpg'), os.path.join(dir_path, dataset, "queries"))

                # Populate data dictionary based on category
                if category in ['ok', 'good', 'junk']:
                    query_info[query_name][category].append(parts[0] + '.jpg')

    # Populate 'gnd' based on query info
    for query_name, info in query_info.items():
        data['gnd'].append(info)

    # Save the result as json
    with open(os.path.join(dir_path, dataset, f'gnd_{dataset}.json'), 'w') as json_file:
        json.dump(data, json_file, indent=4)

    return data
