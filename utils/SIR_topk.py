import os
import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torchvision.io import read_image
from torchvision.utils import make_grid
from PIL import Image


# --- Shared Helpers ---
def add_border(img_tensor, color_name, border_width=10):
    """Draws a colored border around a PyTorch image tensor."""
    colors = {
        'green': torch.tensor([0, 255, 0], dtype=torch.uint8).view(3, 1, 1),
        'red': torch.tensor([255, 0, 0], dtype=torch.uint8).view(3, 1, 1),
        'blue': torch.tensor([0, 150, 255], dtype=torch.uint8).view(3, 1, 1)  # For Query
    }
    color = colors.get(color_name, colors['red'])

    # Clone to avoid modifying the original tensor in memory
    bordered = img_tensor.clone()
    bordered[:, :border_width, :] = color  # Top
    bordered[:, -border_width:, :] = color  # Bottom
    bordered[:, :, :border_width] = color  # Left
    bordered[:, :, -border_width:] = color  # Right
    return bordered


def clean_path(p, cfg):
    """Cleans file paths to match the Ground Truth format for accurate matching."""
    p = p.replace('/', '\\')
    keyword = "queries\\"
    if keyword in p:
        return p[p.find(keyword):]
    root = cfg['dir_data'].replace('/', '\\')
    if p.startswith(root):
        p = p.replace(root, '')
    if p.startswith('.\\'): p = p[2:]
    if p.startswith('\\'): p = p[1:]
    return p


# --- Updated Individual Retrieval Method ---
def retrieve_top_k(cfg, ranks, k, model, retrieve_only=True, save_dir="output"):
    ranks = np.transpose(ranks)
    top_k = {}
    file_path = cfg['path']
    resize_transform = transforms.Resize((224, 224))
    save_dir = os.path.join(file_path, save_dir)

    if not retrieve_only:
        os.makedirs(save_dir, exist_ok=True)

    for i in range(len(ranks)):
        query = cfg['qimlist'][i]
        top_k[query] = {'query': query, 'top_k': []}

        # Get Ground Truth for this specific query
        gnd_item = cfg['gnd'][i]
        expected = set(gnd_item['ok']) | set(gnd_item['good'])

        if not retrieve_only:
            # 1. Load and format the Query Image
            q_img_pil = None
            try:
                q_path = os.path.join(file_path, "queries", query)
                image = read_image(q_path)
                image = resize_transform(image)
                image = add_border(image, 'blue', border_width=12)
                q_img_pil = torchvision.transforms.ToPILImage()(image)
            except Exception as e:
                print(f"Warning: Could not load query image {q_path} - {e}")
                continue

            # 2. Load and score the Top K Results
            result_tensors = []
            for j in range(k):
                next_best = cfg['imlist'][ranks[i][j]]
                best_path = os.path.join(file_path, next_best)
                top_k[query]['top_k'].append(best_path)

                try:
                    # Check against ground truth
                    cleaned_p = clean_path(next_best, cfg)
                    is_correct = cleaned_p in expected

                    image = read_image(best_path)
                    image = resize_transform(image)

                    # Apply Green/Red border
                    border_color = 'green' if is_correct else 'red'
                    image = add_border(image, border_color, border_width=10)

                    result_tensors.append(image)
                except Exception as e:
                    print(f"Warning: Could not load result image {best_path} - {e}")

            # 3. Create the Canvas Layout
            if len(result_tensors) > 0:
                PADDING = 40
                results_grid_tensor = make_grid(torch.stack(result_tensors), nrow=k, padding=5, pad_value=255)
                results_grid_pil = torchvision.transforms.ToPILImage()(results_grid_tensor)

                canvas_width = max(q_img_pil.width, results_grid_pil.width) + (PADDING * 2)
                canvas_height = q_img_pil.height + results_grid_pil.height + (PADDING * 3)
                canvas = Image.new('RGB', (canvas_width, canvas_height), (255, 255, 255))

                # Paste Query
                q_x = (canvas_width - q_img_pil.width) // 2
                canvas.paste(q_img_pil, (q_x, PADDING))

                # Paste Results
                r_x = (canvas_width - results_grid_pil.width) // 2
                canvas.paste(results_grid_pil, (r_x, q_img_pil.height + (PADDING * 2)))

                # Save
                base_file_name = os.path.basename(query)
                short_query_name = os.path.splitext(base_file_name)[0]
                save_path = os.path.join(save_dir, f"{model}_{short_query_name}.jpg")

                canvas.save(save_path)
        else:
            # Just populate the dictionary if retrieve_only is True
            for j in range(k):
                next_best = cfg['imlist'][ranks[i][j]]
                best_path = os.path.join(file_path, next_best)
                top_k[query]['top_k'].append(best_path)

    return top_k


# --- Updated Merged Retrieval Method ---
def save_merged_results(cfg, merged_results, mode, images_per_row=6, save_dir="output"):
    """
    Saves image grids for dynamically sized merged sets (Union/Intersection)
    and exports the raw sets to a .txt file for independent analysis later.
    """
    file_path = cfg['path']
    resize_transform = transforms.Resize((224, 224))
    save_dir = os.path.join(file_path, save_dir)
    os.makedirs(save_dir, exist_ok=True)

    # 1. Export the sets to a TXT file
    txt_filename = os.path.join(save_dir, f"{mode}_raw_sets.txt")
    with open(txt_filename, 'w') as f:
        for query in cfg['qimlist']:
            results = list(merged_results.get(query, []))
            results_str = ",".join(results)
            f.write(f"{query}|{results_str}\n")
    print(f">> Saved {mode} text sets to: {txt_filename}")

    # 2. Generate and Save the Visual Grids
    for i, query in enumerate(cfg['qimlist']):
        result_paths = list(merged_results.get(query, []))
        set_size = len(result_paths)

        # Get Ground Truth for this specific query
        gnd_item = cfg['gnd'][i]
        expected = set(gnd_item['ok']) | set(gnd_item['good'])

        # A. Load the Query Image
        q_img_pil = None
        try:
            q_path = os.path.join(file_path, "queries", query)
            image = read_image(q_path)
            image = resize_transform(image)
            image = add_border(image, 'blue', border_width=12)
            q_img_pil = torchvision.transforms.ToPILImage()(image)
        except Exception as e:
            print(f"Warning: Could not load query image {q_path} - {e}")
            continue

        # B. Load and Score the Results
        result_tensors = []
        for p in result_paths:
            try:
                # Resolve path
                best_path = os.path.join(file_path, p) if not os.path.isabs(p) and not p.startswith(file_path) else p

                # Check against ground truth
                cleaned_p = clean_path(p, cfg)
                is_correct = cleaned_p in expected

                image = read_image(best_path)
                image = resize_transform(image)

                # Apply Green/Red border
                border_color = 'green' if is_correct else 'red'
                image = add_border(image, border_color, border_width=10)

                result_tensors.append(image)
            except Exception as e:
                print(f"Warning: Could not load result {p} - {e}")

        # C. Create the Layout Canvas
        PADDING = 40

        if len(result_tensors) > 0:
            actual_nrow = min(len(result_tensors), images_per_row)
            results_grid_tensor = make_grid(torch.stack(result_tensors), nrow=actual_nrow, padding=5, pad_value=255)
            results_grid_pil = torchvision.transforms.ToPILImage()(results_grid_tensor)
        else:
            # Handle empty intersection sets gracefully
            results_grid_pil = Image.new('RGB', (1, 1), (255, 255, 255))

        canvas_width = max(q_img_pil.width, results_grid_pil.width) + (PADDING * 2)
        canvas_height = q_img_pil.height + results_grid_pil.height + (PADDING * 3)
        canvas = Image.new('RGB', (canvas_width, canvas_height), (255, 255, 255))

        # Paste Query
        q_x = (canvas_width - q_img_pil.width) // 2
        canvas.paste(q_img_pil, (q_x, PADDING))

        # Paste Results
        if len(result_tensors) > 0:
            r_x = (canvas_width - results_grid_pil.width) // 2
            canvas.paste(results_grid_pil, (r_x, q_img_pil.height + (PADDING * 2)))

        # Save
        base_file_name = os.path.basename(query)
        short_query_name = os.path.splitext(base_file_name)[0]
        save_path = os.path.join(save_dir, f"{mode}_{short_query_name}.jpg")

        canvas.save(save_path)