import fiftyone as fo
import os


def fifty_one(data_dir, dataset):
    data = fo.Dataset.from_images_dir(os.path.join(data_dir, dataset))

    session = fo.launch_app(data, desktop=True)
    session.wait()
