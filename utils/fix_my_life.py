import os
import json
import pickle
import glob
import numpy as np


def translate_matrices(snellius_json_path, windows_json_path, pkl_dir):
    print(f"Loading Snellius format: {snellius_json_path}")
    with open(snellius_json_path, 'r') as f:
        s_data = json.load(f)

    print(f"Loading Windows format: {windows_json_path}")
    with open(windows_json_path, 'r') as f:
        w_data = json.load(f)

    # 1. Extract Lists
    s_imlist = s_data['imlist']
    w_imlist = w_data['imlist']

    s_qimlist = [item['query'] for item in s_data['gnd']]
    w_qimlist = [item['query'] for item in w_data['gnd']]

    # 2. Normalize Windows paths to match Linux slashes for the translation
    w_imlist_clean = [img.replace('\\', '/') for img in w_imlist]
    w_qimlist_clean = [q.replace('\\', '/') for q in w_qimlist]

    # 3. Verify Integrity
    print("\nVerifying dataset integrity...")
    assert set(s_imlist) == set(w_imlist_clean), "Error: Gallery image lists do not contain the same files."
    assert set(s_qimlist) == set(w_qimlist_clean), "Error: Query image lists do not contain the same files."
    print("Integrity check passed. File sets match perfectly.")

    # 4. Create Row Permutation (Query Mapping)
    print("\nBuilding translation maps...")
    row_perm = [s_qimlist.index(q) for q in w_qimlist_clean]

    # 5. Create Value Permutation (Gallery Mapping)
    w_imlist_dict = {img: idx for idx, img in enumerate(w_imlist_clean)}
    val_perm = np.zeros(len(s_imlist), dtype=np.int32)
    for s_idx, img in enumerate(s_imlist):
        val_perm[s_idx] = w_imlist_dict[img]

    # 6. Process PKL Files
    pkl_files = glob.glob(os.path.join(pkl_dir, '*_data.pkl'))
    if not pkl_files:
        print(f"\nNo .pkl files found in {pkl_dir}")
        return

    print("\nTranslating PKL matrices...")
    for pkl_file in pkl_files:
        print(f" -> Processing: {os.path.basename(pkl_file)}")
        with open(pkl_file, 'rb') as f:
            pool = pickle.load(f)

        for key, info in pool.items():
            if 'ranks' in info:
                old_ranks = info['ranks']

                # Ensure it's a numpy array for safe indexing
                if hasattr(old_ranks, 'cpu'):
                    old_ranks = old_ranks.cpu().numpy()
                elif isinstance(old_ranks, list):
                    old_ranks = np.array(old_ranks)

                # Apply Query Reordering to the COLUMNS
                new_ranks = old_ranks[:, row_perm]

                # Apply Gallery Remapping to the matrix values
                new_ranks = val_perm[new_ranks]

                info['ranks'] = new_ranks

        # Backup original and save translated file
        backup_path = pkl_file + ".bak"
        if not os.path.exists(backup_path):
            os.rename(pkl_file, backup_path)
            print("    (Original file backed up safely)")
        else:
            print("    (Backup already exists, overwriting current .pkl)")

        with open(pkl_file, 'wb') as f:
            pickle.dump(pool, f)

    print("\nSUCCESS: All matrices translated to Windows order.")


if __name__ == "__main__":
    # Ensure this points to your ILIAS dataset folder
    DATA_DIR = r"C:\Users\brand\Documents\TUe Master\Thesis\SimilarityImageRetrieval\datasets\ILIAS"

    snellius_json = os.path.join(DATA_DIR, "gnd_ILIAS_linux.json")
    windows_json = os.path.join(DATA_DIR, "gnd_ILIAS.json")
    pkl_directory = os.path.join(DATA_DIR, "checkpoints")

    translate_matrices(snellius_json, windows_json, pkl_directory)