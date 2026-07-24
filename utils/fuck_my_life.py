import os
import pickle


def verify_pkl(file_path):
    if not os.path.exists(file_path):
        print(f"[!] File not found: {file_path}")
        return

    print(
        f"\n--- Checking: {os.path.basename(file_path)} ({os.path.getsize(file_path) / (1024 * 1024 * 1024):.2f} GB) ---")

    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        print(f"Total Models/Combinations inside: {len(data.keys())}")

        # Sample the first combination to check the matrix shape
        first_key = list(data.keys())[0]
        ranks = data[first_key].get('ranks')

        if hasattr(ranks, 'shape'):
            print(f"Sample '{first_key}' Matrix Shape: {ranks.shape} | Type: {type(ranks)}")
        else:
            print(f"Sample '{first_key}' Matrix Length: {len(ranks)} rows | Type: Nested List")

    except Exception as e:
        print(f"[!] CORRUPTED FILE: {e}")


if __name__ == "__main__":
    # Point this to your checkpoints directory
    DATA_DIR = r"C:\Users\brand\Documents\TUe Master\Thesis\SimilarityImageRetrieval\datasets\ILIAS\checkpoints"

    # Check the original 30GB Snellius file
    verify_pkl(os.path.join(DATA_DIR, "dino_data.pkl.bak"))

    # Check the new 500MB Translated file
    verify_pkl(os.path.join(DATA_DIR, "dino_data.pkl"))