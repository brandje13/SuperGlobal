import os


def rename_images(directory):
    print(directory)
    name = directory.split("/")[-2] + "_"
    print(name)
    for filename in os.listdir(directory):
        if (filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg")) and not filename.startswith(name):
            os.rename(os.path.join(directory, filename), os.path.join(directory, name + filename))
            print(filename)

rename_images("./revisitop/catndogs/PetImages/Dog/")