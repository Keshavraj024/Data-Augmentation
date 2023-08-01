import numpy as np
from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from tqdm import tqdm
from pathlib import Path


class DataAugmentation:
    def __init__(self, batch: int, img_path: str, num_images: int, save_dir: str):
        """Data Augmentation class.

		Args:
			batch (int): Number of images to generate in each batch
			img_path (str): File path of the input image
			num_images (int): Total number of augmented images to generate
			save_dir (str): Directory to save the augmented images
		"""
        self.batch = batch
        self.img_path = img_path
        self.num_images = num_images
        self.save_dir = save_dir
        self.generate_augmented_images()

    def generate_augmented_images(self) -> None:
        """
        Generate augmented images and save them to the specified directory.
        """
        img = load_img(self.img_path)
        img = img_to_array(img)
        img = np.expand_dims(img, 0)

        path = Path(self.save_dir)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)

        datagen = ImageDataGenerator(
            vertical_flip=True,
            horizontal_flip=True,
            brightness_range=[0.7, 1.0],
            rotation_range=135,
            zoom_range=[0.2, 0.7],
            width_shift_range=0.2,
            height_shift_range=0.2,
        )

        image_count = 0
        for _ in tqdm(
            datagen.flow(
                img,
                batch_size=self.batch,
                save_to_dir=self.save_dir,
                save_prefix="image",
                save_format="jpeg",
            )
        ):
            image_count += 1
            if image_count >= self.num_images:
                break


if __name__ == "__main__":
    DataAugmentation(batch=1, img_path="Image/dog.jpg", num_images=2, save_dir="Augmented_Data")
