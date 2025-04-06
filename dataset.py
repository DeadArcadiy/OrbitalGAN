import glob
import torch
from PIL import Image
from torch.utils.data import Dataset
import os
import urllib
import zipfile
from tqdm import tqdm

class Maps(Dataset):
    url="https://github.com/akanametov/pix2pix/releases/download/1.0/maps.zip"
    def __init__(self,
                 root: str='.',
                 transform=None,
                 download: bool=True,
                 mode: str='train',
                 direction: str='A2B'):
        if download:
            _ = download_and_extract(root, self.url)
        self.root=root
        self.files=sorted(glob.glob(f"{root}/maps/{mode}/*.jpg"))
        self.transform=transform
        self.download=download
        self.mode=mode
        self.direction=direction
        
    def __len__(self,):
        return len(self.files)
    
    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert('RGB')
        W, H = img.size
        cW = W//2
        imgA = img.crop((0, 0, cW, H))
        imgB = img.crop((cW, 0, W, H))
        
        if self.transform:
            imgA, imgB = self.transform(imgA, imgB)
            
        if self.direction == 'A2B':
            return imgA, imgB
        else:
            return imgB, imgA
        
def download(url: str, filename: str, chunk_size: int = 4096) -> None:
    with open(filename, "wb") as fh:
        with urllib.request.urlopen(urllib.request.Request(url)) as response:
            with tqdm(total=response.length) as pbar:
                for chunk in iter(lambda: response.read(chunk_size), ""):
                    if not chunk:
                        break
                    pbar.update(chunk_size)
                    fh.write(chunk)
    print(f'Dataset downloaded!')
    return None

def extract(from_path: str, to_path: str) -> None:
    with zipfile.ZipFile(from_path, "r", compression=zipfile.ZIP_STORED) as zf:
        zf.extractall(to_path)
    print('Dataset extracted!')
    return None

def download_and_extract(root: str, url: str, filename: str=None):
    root = os.path.expanduser(root)
    if not filename:
        filename = os.path.basename(url)
    fpath = os.path.join(root, filename)
    if os.path.exists(fpath):
        print('Dataset is already downloaded!')
    else:
        os.makedirs(root, exist_ok=True)
        _ = download(url, fpath)
        _ = extract(fpath, root)
    return None