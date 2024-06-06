import fiftyone as fo
import fiftyone.zoo as foz

dataset = fo.Dataset.from_images_dir(".\\revisitop\\roxford5k")

session = fo.launch_app(dataset, desktop=True)
session.wait()