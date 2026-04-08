import cv2
from torchvision import transforms
from dataloader.dataset import DataSet
from config import cfg as c


class DataSet_ConvNeXtV2(DataSet):
    def __init__(self, data_path, dataset, fn, split):
        resolution = c.ConvNeXtV2.RESOLUTION

        # Standard ImageNet normalization used by timm backbones
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((resolution, resolution), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225)),
        ])

        # Calling super() automatically triggers the parent class's flawless _construct_db
        super().__init__(data_path, dataset, fn, split)

    def __getitem__(self, index):
        # The parent class correctly parses the JSON, handles the paths, and crops the bounding boxes
        im = self._load_img(index)

        # Failsafe so we get a readable Python error instead of a cryptic C++ OpenCV crash
        if im is None:
            raise ValueError(f"OpenCV failed to load image at index {index}. The base DataSet constructed a bad path.")

        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im_tensor = self.transform(im)
        return im_tensor