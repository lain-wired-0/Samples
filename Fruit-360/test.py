from imutils import paths
import random
from keras.preprocessing.image import img_to_array


train_image_paths = sorted(list(paths.list_images('fruits360-kaggle/dataset/fruits-360/Training')))
test_image_paths = sorted(list(paths.list_images('fruits360-kaggle/dataset/fruits-360/Test')))
print(len(train_image_paths))

background_image_paths = sorted(list(paths.list_images('fruits360-kaggle/dataset/Backgrounds')))
to_delete = random.sample(range(len(background_image_paths)), 500)
train_background_image_paths = [background_image_paths.pop(i) for i, _ in reversed(list(enumerate(background_image_paths))) if i in to_delete]
to_delete = random.sample(range(len(background_image_paths)), 160)
test_background_image_paths = [background_image_paths.pop(i) for i, _ in reversed(list(enumerate(background_image_paths))) if i in to_delete]
print(len(train_background_image_paths))

train_image_paths = list(train_image_paths) + train_background_image_paths
# list(test_image_paths).extend(test_background_image_paths)

# print(train_image_paths[0])
print(test_image_paths[0])
