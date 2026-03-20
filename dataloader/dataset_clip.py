import os
import json
import cv2
from torchvision import transforms
from dataloader.dataset import DataSet


class DataSet_CLIP(DataSet):
    """CLIP dataset handling Text for queries and Images for the database."""

    def __init__(self, data_path, dataset, fn, split):
        # We define the transforms BEFORE calling super() because super()
        # will immediately trigger _construct_db, which we are overriding.
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                 std=(0.26862954, 0.26130258, 0.27577711)),
        ])

        # This automatically sets self._split and calls our custom _construct_db
        super().__init__(data_path, dataset, fn, split)

    def _construct_db(self):
        """Override the parent's DB construction to handle text queries."""
        self._db = []

        # Open the JSON file exactly like the parent class does
        with open(os.path.join(self._data_path, self._dataset, self._fn), 'rb') as fin:
            gnd = json.load(fin)

            # --- TEXT LOGIC (For Queries) ---
            if self._split == "query":
                # Back to iterating through the standard qimlist
                for i in range(len(gnd["qimlist"])):
                    # Grab the injected text, defaulting to "" if missing
                    text_str = gnd["gnd"][i].get("text", "")
                    self._db.append({"text": text_str})

            # --- IMAGE LOGIC (For Database) ---
            elif self._split == "db":
                for i in range(len(gnd["imlist"])):
                    im_fn = gnd["imlist"][i]
                    im_path = os.path.join(self._data_path, self._dataset, im_fn)
                    self._db.append({"im_path": im_path})

    def __getitem__(self, index):
        # 1. TEXT REQUEST: For query features
        if self._split == 'query':
            return self._db[index]["text"]

        # 2. IMAGE REQUEST: For dataset/database features
        elif self._split == 'db':
            # Rely on the parent class's image loader
            im = self._load_img(index)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im_tensor = self.transform(im)
            return im_tensor