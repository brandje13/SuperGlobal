import os
import shutil
from PIL import Image
import json


def create_groundtruth(query_paths, dir_path, dataset):
    data = {'imlist': [], 'qimlist': [], 'gnd': [], 'path': str}
    query_info = {}
    data['path'] = os.path.join(dir_path, dataset)

    # Iterate through each file in the directory
    for img in sorted(os.listdir(os.path.join(dir_path, dataset))):
        if img.endswith((".jpg", ".png", ".jpeg")):
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
                    pos_file = os.path.join("queries", os.path.basename(path), 'pos', file)
                    data['imlist'].append(pos_file)
                    class_pos_images.append(pos_file)

            # B. Find the text description for this class
            class_text = ""
            for file in sorted(os.listdir(temp_path_query)):
                if file.endswith(".txt") and file.startswith("T"):
                    with open(os.path.join(temp_path_query, file), 'r', encoding='utf-8') as f:
                        class_text = f.read().strip()
                    break

            # C. Process Image Queries and inject the text
            for file in sorted(os.listdir(temp_path_query)):
                if file.endswith((".jpg", ".png", ".jpeg")):
                    query_file = os.path.join(os.path.basename(path), 'query', file)
                    query_name = os.path.splitext(file)[0]
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
                            'text': class_text,
                            'ok': class_pos_images.copy(),
                            'good': [],
                            'junk': []
                        }
                        data['qimlist'].append(query_file)
    else:
        # Iterate over all image files in the directory
        for filename in sorted(query_paths):
            query_name = os.path.basename(filename)
            category = 'query'

            if query_name not in query_info:
                query_info[query_name] = {'query': None, 'bbx': None, 'text': "",
                                          'ok': [], 'good': [], 'junk': []}

            if category == 'query':
                query_info[query_name][category] = query_name
                w, h = Image.open(filename).size
                query_info[query_name]['bbx'] = [0, 0, w, h]
                data['qimlist'].append(query_name)

            if category in ['ok', 'good', 'junk']:
                query_info[query_name][category].append(None)

    for query_name, info in query_info.items():
        data['gnd'].append(info)

    with open(os.path.join(dir_path, dataset, f'gnd_{dataset}.json'), 'w') as json_file:
        json.dump(data, json_file, indent=4)


def create_groundtruth_from_txt(dir_path, dataset):
    data = {'imlist': [], 'qimlist': [], 'gnd': [], 'path': str}
    query_info = {}
    data['path'] = os.path.join(dir_path, dataset)

    for img in sorted(os.listdir(os.path.join(dir_path, dataset))):
        if img.endswith((".jpg", ".png", ".jpeg")):
            data['imlist'].append(img)

    groundtruth_dir = os.path.join(dir_path, dataset, "groundtruth")
    for filename in sorted(os.listdir(groundtruth_dir)):
        if filename.endswith('.txt'):
            parts = filename.split('_')
            query_name = '_'.join(parts[:-1])
            category = parts[-1][:-4]

            with open(os.path.join(groundtruth_dir, filename), 'r') as file:
                lines = file.readlines()

            for line in lines:
                parts = line.split()
                if query_name not in query_info:
                    query_info[query_name] = {'query': None, 'bbx': None, 'text': "",
                                              'ok': [], 'good': [], 'junk': []}

                if category == 'query':
                    file_name = parts[0][5:] + '.jpg'
                    query_info[query_name][category] = file_name
                    query_info[query_name]['bbx'] = list(map(float, parts[1:]))
                    data['qimlist'].append(file_name)

                    src_path = os.path.join(dir_path, dataset, file_name)
                    dst_path = os.path.join(dir_path, dataset, "queries", file_name)
                    shutil.copy(src_path, dst_path)

                if category in ['ok', 'good', 'junk']:
                    query_info[query_name][category].append(parts[0] + '.jpg')

    for query_name, info in query_info.items():
        data['gnd'].append(info)

    with open(os.path.join(dir_path, dataset, f'gnd_{dataset}.json'), 'w') as json_file:
        json.dump(data, json_file, indent=4)

    return data