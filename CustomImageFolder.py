from torchvision.datasets import VisionDataset
import torchvision.transforms as transforms
import pandas as pd
from torchvision.datasets.folder import default_loader

diseases = ["esantema-virale", "esantema-maculo-papuloso", "scabbia"]

class CustomImageFolderColor(VisionDataset):
    def __init__(self, root, transform=transforms.ToTensor(), target_transform=None):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(diseases)}
        self.imgs = self._load_imgs()

    def _load_imgs(self):
        # here, I first create a dictionary which associates each image path to its skin label. After that, I incorporate this label in my data.

        imgs = []
        for disease in diseases:
          df = pd.read_csv(f"{self.root}/skin_color_datasets/{disease}_ITA.csv")
          skin_labels = df["skin_label"]
          img_paths = df["img_path"]

          imgs.extend([("../Thesis/"+img.replace("\\", "/"), skin_label, self.class_to_idx[disease]) for img, skin_label in zip(img_paths, skin_labels)])

        return imgs

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path, skin_label, target = self.imgs[idx]
        img = default_loader(img_path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, skin_label, target

class CustomImageFolderColorGen(VisionDataset):
    def __init__(self, root, FLAG, gen_black=[0]*9, gen_brown=[0]*9, transform=transforms.ToTensor(), target_transform=None):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(diseases)}
        self.gen_black = gen_black
        self.gen_brown = gen_brown
        self.imgs = self._load_imgs()

    def _load_imgs(self):
        imgs = []
        for disease in diseases:

            disease_idx = self.class_to_idx[disease]
            num_black = self.gen_black[disease_idx]
            num_brown = self.gen_brown[disease_idx]
            black_files = [os.path.join(f"colab/DB/{disease}/{FLAG}/black_gen/", f) for f in os.listdir(f"colab/DB/{disease}/{FLAG}/black_gen") if f.lower().endswith('.png')]
            brown_files = [os.path.join(f"colab/DB/{disease}/{FLAG}/brown_gen/", f) for f in os.listdir(f"colab/DB/{disease}/{FLAG}/brown_gen") if f.lower().endswith('.png')]

            random.seed(42)
            random.shuffle(black_files)
            random.shuffle(brown_files)

            black_files = black_files[:num_black]
            brown_files = brown_files[:num_brown]

            imgs.extend([(img, "dark", self.class_to_idx[disease]) for img in black_files])
            imgs.extend([(img, "brown", self.class_to_idx[disease]) for img in brown_files])

        return imgs

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path, skin_label, target = self.imgs[idx]
        img = default_loader(img_path).resize((256,256))
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, skin_label, target