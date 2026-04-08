import cv2
from torchvision import transforms
from dataloader.dataset import DataSet
from config import cfg as c


class DataSet_DINO(DataSet):
    """DINOv2 dataset with Dynamic resolution."""

    def __init__(self, data_path, dataset, fn, split):
        super().__init__(data_path, dataset, fn, split)

        resolution = c.DINO.RESOLUTION

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((resolution, resolution), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    def __getitem__(self, index):
        im = self._load_img(index)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im_tensor = self.transform(im)
        return im_tensor