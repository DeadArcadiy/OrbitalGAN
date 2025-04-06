import os
import random
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

# Трансформация для случайного обрезания пары изображений
class RandomCropPair(object):
    def __init__(self, size):
        self.size = size  # Размер обрезки (ширина, высота)

    def __call__(self, in_image, out_image):
        # Получаем размеры входного изображения
        w, h = in_image.size
        
        # Генерируем случайные координаты для верхнего левого угла прямоугольника
        top = random.randint(0, h - self.size[1])
        left = random.randint(0, w - self.size[0])
        
        # Обрезаем оба изображения с одинаковыми координатами
        in_image = in_image.crop((left, top, left + self.size[0], top + self.size[1]))
        out_image = out_image.crop((left, top, left + self.size[0], top + self.size[1]))

        return in_image, out_image

class ImagePairDataset(Dataset):
    def __init__(self, folder_in, folder_out, transform_in=None, transform_out=None, crop_size=(256, 256)):
        """
        :param folder_in: Папка с входными изображениями.
        :param folder_out: Папка с выходными изображениями.
        :param transform_in: Трансформации для входных изображений.
        :param transform_out: Трансформации для выходных изображений.
        :param crop_size: Размер обрезки.
        """
        self.folder_in = folder_in
        self.folder_out = folder_out
        self.transform_in = transform_in
        self.transform_out = transform_out

        # Инициализируем RandomCropPair
        self.random_crop = RandomCropPair(crop_size)

        # Получаем список всех файлов в папках
        self.in_files = sorted(os.listdir(folder_in))
        self.out_files = sorted(os.listdir(folder_out))

        # Фильтруем только те файлы, которые есть в обеих папках
        self.pairs = [file for file in self.in_files if file in self.out_files]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        # Получаем пару изображений
        in_file = self.pairs[idx]
        out_file = in_file  # У нас одинаковые имена файлов для входных и выходных данных

        in_image = Image.open(os.path.join(self.folder_in, in_file))
        out_image = Image.open(os.path.join(self.folder_out, out_file))

        # Конвертируем в RGB, чтобы удалить альфа-канал, если он есть
        in_image = in_image.convert('RGB')
        out_image = out_image.convert('RGB')

        # Применяем случайное обрезание к обеим картинкам
        in_image, out_image = self.random_crop(in_image, out_image)

        # Применяем другие трансформации (если они есть)
        if self.transform_in:
            in_image = self.transform_in(in_image)
        if self.transform_out:
            out_image = self.transform_out(out_image)

        return in_image, out_image
