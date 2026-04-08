import os
import json
import cv2
from torchvision import transforms
from dataloader.dataset import DataSet
from config import cfg as c

class DataSet_SigLIP(DataSet):
    def __init__(self, data_path, dataset, fn, split):
        resolution = c.SigLIP.RESOLUTION

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((resolution, resolution), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                 std=(0.5, 0.5, 0.5)),
        ])

        super().__init__(data_path, dataset, fn, split)

    def _construct_db(self):
        self._db = []

        with open(os.path.join(self._data_path, self._dataset, self._fn), 'rb') as fin:
            gnd = json.load(fin)

            if self._split == "query":
                for i in range(len(gnd["qimlist"])):
                    text_str = gnd["gnd"][i].get("text", "")
                    self._db.append({"text": text_str})

            elif self._split == "db":
                for i in range(len(gnd["imlist"])):
                    im_fn = gnd["imlist"][i]
                    im_path = os.path.join(self._data_path, self._dataset, im_fn)
                    self._db.append({"im_path": im_path})

    def __getitem__(self, index):
        if self._split == 'query':
            return self._db[index]["text"]

        elif self._split == 'db':
            im = self._load_img(index)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im_tensor = self.transform(im)
            return im_tensor