import os
import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torchvision.io import read_image
from torchvision.utils import make_grid
from PIL import Image, ImageDraw, ImageFont


# --- Shared Helpers ---
def add_border(img_tensor, color_name, border_width=10):
    """Draws a colored border around a PyTorch image tensor."""
    colors = {
        'green': torch.tensor([0, 255, 0], dtype=torch.uint8).view(3, 1, 1),
        'red': torch.tensor([255, 0, 0], dtype=torch.uint8).view(3, 1, 1),
        'blue': torch.tensor([0, 150, 255], dtype=torch.uint8).view(3, 1, 1)  # For Query
    }
    color = colors.get(color_name, colors['red'])

    bordered = img_tensor.clone()
    bordered[:, :border_width, :] = color  # Top
    bordered[:, -border_width:, :] = color  # Bottom
    bordered[:, :, :border_width] = color  # Left
    bordered[:, :, -border_width:] = color  # Right
    return bordered


def add_source_border(img_tensor, is_correct, model_hits, border_width=10):
    """
    Draws a base Ground Truth border, then dynamically colors specific edges
    based on which models retrieved the image.
    """
    bordered = img_tensor.clone()

    # Base Ground Truth Colors
    c_correct = torch.tensor([0, 255, 0], dtype=torch.uint8).view(3, 1, 1)  # Green
    c_incorrect = torch.tensor([255, 0, 0], dtype=torch.uint8).view(3, 1, 1)  # Red
    base_color = c_correct if is_correct else c_incorrect

    # 1. Paint the entire border with the Ground Truth color first
    bordered[:, :border_width, :] = base_color  # Top
    bordered[:, -border_width:, :] = base_color  # Bottom
    bordered[:, :, :border_width] = base_color  # Left
    bordered[:, :, -border_width:] = base_color  # Right

    # Generalized palette (supports many models before looping colors)
    palette = [
        [128, 0, 128],  # Purple
        [0, 150, 255],  # Blue
        [255, 165, 0],  # Orange
        [0, 255, 255],  # Cyan
        [255, 0, 255],  # Magenta
        [255, 255, 0]  # Yellow
    ]

    edges = ['left', 'right', 'top', 'bottom']

    for i, hit in enumerate(model_hits):
        if hit:
            color_val = palette[i % len(palette)]
            c_model = torch.tensor(color_val, dtype=torch.uint8).view(3, 1, 1)

            edge = edges[i % 4]

            if edge == 'left':
                bordered[:, :, :border_width] = c_model
            elif edge == 'top':
                bordered[:, :border_width, :] = c_model
            elif edge == 'right':
                bordered[:, :, -border_width:] = c_model
            elif edge == 'bottom':
                bordered[:, -border_width:, :] = c_model

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


def draw_stats_block(canvas, x_offset, y_offset, stats_dict, line_spacing=30):
    """Draws a clean text block of statistics onto the PIL canvas."""
    draw = ImageDraw.Draw(canvas)

    # Dropped font size to 22 for a tighter look
    try:
        font = ImageFont.truetype("arial.ttf", 22)
    except IOError:
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 22)
        except IOError:
            font = ImageFont.load_default()

    y_text = y_offset
    for key, value in stats_dict.items():
        text_line = f"{key}: {value}"
        # Black text
        draw.text((x_offset, y_text), text_line, font=font, fill=(0, 0, 0))
        y_text += line_spacing  # Uses the new tighter spacing


# --- Updated Individual Retrieval Method ---
def retrieve_top_k(cfg, ranks, k, model, retrieve_only=True, save_dir="output", images_per_row=5):
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

        gnd_item = cfg['gnd'][i]
        expected = set(gnd_item['ok']) | set(gnd_item['good'])
        total_available = len(expected)

        if not retrieve_only:
            # 1. Load Query Image
            q_img_pil = None
            try:
                q_path = os.path.join(file_path, "queries", query)
                image = read_image(q_path)
                image = resize_transform(image)
                # FIXED: Border width is 10 to perfectly match the grid images
                image = add_border(image, 'blue', border_width=10)
                q_img_pil = torchvision.transforms.ToPILImage()(image)
            except Exception as e:
                print(f"Warning: Could not load query image {q_path} - {e}")
                continue

            # 2. Load and score Results
            result_tensors = []
            true_positives = 0
            ap_sum = 0.0

            # Cap j so it never exceeds the array bounds
            actual_k = min(k, len(ranks[i]))
            for j in range(actual_k):
                next_best = cfg['imlist'][ranks[i][j]]
                best_path = os.path.join(file_path, next_best)
                top_k[query]['top_k'].append(best_path)

                try:
                    cleaned_p = clean_path(next_best, cfg)
                    is_correct = cleaned_p in expected

                    if is_correct:
                        true_positives += 1
                        ap_sum += (true_positives / (j + 1))

                    image = read_image(best_path)
                    image = resize_transform(image)
                    border_color = 'green' if is_correct else 'red'
                    image = add_border(image, border_color, border_width=10)

                    result_tensors.append(image)
                except Exception as e:
                    print(f"Warning: Could not load result image {best_path} - {e}")

            average_precision = (ap_sum / total_available) if total_available > 0 else 0.0

            # --- FIXED: Pad the grid even if 0 results are found ---
            if len(result_tensors) < images_per_row:
                num_empty = images_per_row - len(result_tensors)
                empty_shape = (3, 224, 224)
                empty_tensor = torch.ones(empty_shape, dtype=torch.uint8) * 255
                for _ in range(num_empty):
                    result_tensors.append(empty_tensor)

            # 3. Create Canvas and draw grid
            PADDING = 40
            GRID_PAD = 5

            # Lock nrow to images_per_row
            results_grid_tensor = make_grid(torch.stack(result_tensors), nrow=images_per_row, padding=GRID_PAD,
                                            pad_value=255)
            results_grid_pil = torchvision.transforms.ToPILImage()(results_grid_tensor)

            min_query_area_width = q_img_pil.width + 350
            canvas_width = max(min_query_area_width, results_grid_pil.width) + (PADDING * 2)
            canvas_height = q_img_pil.height + results_grid_pil.height + (PADDING * 3)
            canvas = Image.new('RGB', (canvas_width, canvas_height), (255, 255, 255))

            # --- Pixel-Perfect Alignment Math ---
            # Calculate Grid Position FIRST
            r_x = (canvas_width - results_grid_pil.width) // 2

            # Lock Query X to the Grid's internal image start
            q_x = r_x + GRID_PAD
            canvas.paste(q_img_pil, (q_x, PADDING))

            stats = {
                "Model": model,
                "Total Retrieved (K)": k,
                "True Positives Found": true_positives,
                "Total Positives in DB": total_available,
                "Recall": f"{(true_positives / total_available) * 100:.1f}%" if total_available > 0 else "0.0%",
                "Precision": f"{(true_positives / k) * 100:.1f}%",
                "Average Precision (AP)": f"{average_precision * 100:.1f}%"
            }

            # Calculate how tall the text block will be
            line_spacing = 30
            total_text_height = len(stats) * line_spacing

            # Push it 40 pixels to the right of the query image
            stats_x = q_x + q_img_pil.width + 40

            # Vertically center it against the query image's height
            stats_y = PADDING + ((q_img_pil.height - total_text_height) // 2)

            # Draw the text using the dynamic coordinates
            draw_stats_block(canvas, stats_x, stats_y, stats, line_spacing)

            # Paste Results Grid Below
            canvas.paste(results_grid_pil, (r_x, q_img_pil.height + (PADDING * 2)))

            # Save
            base_file_name = os.path.basename(query)
            short_query_name = os.path.splitext(base_file_name)[0]
            save_path = os.path.join(save_dir, f"{model}_{short_query_name}.jpg")
            canvas.save(save_path)
        else:
            actual_k = min(k, len(ranks[i]))
            for j in range(actual_k):
                next_best = cfg['imlist'][ranks[i][j]]
                best_path = os.path.join(file_path, next_best)
                top_k[query]['top_k'].append(best_path)

    return top_k


# --- Updated Merged Retrieval Method ---
def save_merged_results(cfg, merged_results, models, mode, images_per_row=5, save_dir="output"):
    """
    Saves image grids for dynamically sized merged sets (Union/Intersection)
    and exports the raw sets to a .txt file for independent analysis later.
    """

    # --- COLOR MODE SWITCH ---
    # Set to False to see colorful model edge tracking (diagnostic).
    # Set to True for solid Green/Red academic report figures.
    color_mode = False
    # ------------------------------------

    file_path = cfg['path']
    resize_transform = transforms.Resize((224, 224))
    save_dir = os.path.join(file_path, save_dir)
    os.makedirs(save_dir, exist_ok=True)

    txt_filename = os.path.join(save_dir, f"{mode}_raw_sets.txt")
    with open(txt_filename, 'w') as f:
        for query in cfg['qimlist']:
            results = list(merged_results.get(query, []))
            results_str = ",".join(results)
            f.write(f"{query}|{results_str}\n")
    print(f">> Saved {mode} text sets to: {txt_filename}")

    for i, query in enumerate(cfg['qimlist']):
        result_paths = list(merged_results.get(query, []))
        set_size = len(result_paths)

        gnd_item = cfg['gnd'][i]
        expected = set(gnd_item['ok']) | set(gnd_item['good'])
        total_available = len(expected)

        q_img_pil = None
        try:
            q_path = os.path.join(file_path, "queries", query)
            image = read_image(q_path)
            image = resize_transform(image)
            image = add_border(image, 'blue', border_width=10)
            q_img_pil = torchvision.transforms.ToPILImage()(image)
        except Exception as e:
            print(f"Warning: Could not load query image {q_path} - {e}")
            continue

        result_tensors = []
        true_positives = 0

        for p in result_paths:
            try:
                best_path = os.path.join(file_path, p) if not os.path.isabs(p) and not p.startswith(file_path) else p
                cleaned_p = clean_path(p, cfg)
                is_correct = cleaned_p in expected

                if is_correct:
                    true_positives += 1

                image = read_image(best_path)
                image = resize_transform(image)

                # --- APPLY BORDER BASED ON VISUALIZATION MODE ---
                if not color_mode:
                    # Academic Reporting: Simple Green/Red solid borders
                    border_color = 'green' if is_correct else 'red'
                    image = add_border(image, border_color, border_width=10)
                else:
                    # Diagnostic (Rainbow) Mode: Tracking edges for model source
                    # --- GENERALIZED MODEL CHECK ---
                    model_hits = [p in m[1][query]['top_k'] for m in models]
                    image = add_source_border(image, is_correct, model_hits, border_width=10)

                result_tensors.append(image)
            except Exception as e:
                pass

        # --- FIXED: Pad the grid even if 0 results are found ---
        if len(result_tensors) < images_per_row:
            num_empty = images_per_row - len(result_tensors)
            empty_shape = (3, 224, 224)
            empty_tensor = torch.ones(empty_shape, dtype=torch.uint8) * 255
            for _ in range(num_empty):
                result_tensors.append(empty_tensor)

        PADDING = 40
        GRID_PAD = 5

        # Lock nrow to images_per_row
        results_grid_tensor = make_grid(torch.stack(result_tensors), nrow=images_per_row, padding=GRID_PAD,
                                        pad_value=255)
        results_grid_pil = torchvision.transforms.ToPILImage()(results_grid_tensor)

        # --- Split into TWO Columns: Stats and Legend ---
        stats = {
            "Merge Mode": mode.capitalize(),
            "Total Shown (K)": set_size,
            "True Positives Found": true_positives,
            "Total Positives in DB": total_available,
            "Recall": f"{(true_positives / total_available) * 100:.1f}%" if total_available > 0 else "0.0%",
            "Precision": f"{(true_positives / set_size) * 100:.1f}%" if set_size > 0 else "0.0%"
        }

        # Define the Legend dynamically based on Thesis Reporting Mode
        if not color_mode:
            # academic legend
            legend = {
                "FIGURE LEGEND": "",
                "Green Border": "Matches Ground Truth",
                "Red Border": "Distractor Image"
            }
        else:
            # diagnostic edge tracking legend
            legend = {
                "EDGE LEGEND": "",
                "Base (Green/Red)": "Ground Truth "
            }

            edges_names = ['Left', 'Right', 'Top', 'Bottom']
            color_names = ['Purple', 'Orange', 'Blue', 'Cyan', 'Magenta', 'Yellow']  # Adjusted names list to match

            # Dynamically fill the legend with the models present
            for idx, m in enumerate(models):
                model_name = m[0]
                edge = edges_names[idx % 4]
                color = color_names[idx % len(color_names)]
                legend[f"{edge} Edge ({color})"] = model_name

        # --- Layout Math for Two Columns ---
        line_spacing = 30

        # Determine the tallest element to define the top section height
        max_lines = max(len(stats), len(legend))
        total_text_height = max_lines * line_spacing
        top_section_height = max(q_img_pil.height, total_text_height)

        # We need extra width for the second column (~320px per column)
        min_query_area_width = q_img_pil.width + 660
        canvas_width = max(min_query_area_width, results_grid_pil.width) + (PADDING * 2)
        canvas_height = top_section_height + results_grid_pil.height + (PADDING * 3)
        canvas = Image.new('RGB', (canvas_width, canvas_height), (255, 255, 255))

        # Calculate X positions
        r_x = (canvas_width - results_grid_pil.width) // 2
        q_x = r_x + GRID_PAD

        # Position the two text columns
        stats_x = q_x + q_img_pil.width + 40
        legend_x = stats_x + 320  # Pushes the legend column to the right

        # Center everything vertically in the top section
        q_y = PADDING + ((top_section_height - q_img_pil.height) // 2)
        text_y = PADDING + ((top_section_height - total_text_height) // 2)

        canvas.paste(q_img_pil, (q_x, q_y))

        # Draw BOTH text blocks
        draw_stats_block(canvas, stats_x, text_y, stats, line_spacing)
        draw_stats_block(canvas, legend_x, text_y, legend, line_spacing)

        # Paste the grid safely below the top section
        grid_y = top_section_height + (PADDING * 2)
        canvas.paste(results_grid_pil, (r_x, grid_y))

        base_file_name = os.path.basename(query)
        short_query_name = os.path.splitext(base_file_name)[0]
        save_path = os.path.join(save_dir, f"{mode}_{short_query_name}.jpg")
        canvas.save(save_path)