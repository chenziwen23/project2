import torch, glob
from torch.utils.data import DataLoader, Dataset
from tqdm.notebook import tqdm
import torchvision.transforms as transforms
from PIL import Image
import albumentations
from turbojpeg import TurboJPEG


abdir = '/mnt/chenziwen/dc/train/*.jpg'


class BasicDataset(Dataset):
    def __init__(self, transform=None):
        self.img_list = glob.glob(abdir)
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        image = Image.open(self.img_list[idx])

        if self.transform is not None:
            image = self.transform(image)

        return image


class jpeg4pyDataset(Dataset):
    def __init__(self, transform=None):
        self.img_list = glob.glob(abdir)
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        image = jpeg.JPEG(self.img_list[idx]).decode()
        image = Image.fromarray(image)

        if self.transform is not None:
            image = self.transform(image)

        return image
    

class jpeg4pyalbDataset(Dataset):
    def __init__(self, transform=None):
        self.img_list = glob.glob(abdir)
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        image = jpeg.JPEG(self.img_list[idx]).decode()

        if self.transform is not None:
            image = self.transform(**{"image": image})

        return torch.from_numpy(image['image'])

    
class turbojpeg4pyalbDataset(Dataset):
    def __init__(self, transform=None):
        self.img_list = glob.glob(abdir)
        self.transform = transform
        self.jpeg = TurboJPEG()
        
    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        image = self.jpeg.decode(open(self.img_list[idx], 'rb').read())

        if self.transform is not None:
            image = self.transform(**{"image": image})

        return torch.from_numpy(image['image'])
    
transform1 = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

transform2 = albumentations.Compose([
                                albumentations.Resize(height=224, width=224, interpolation=1, always_apply=True, p=1),
                                albumentations.Flip(always_apply=False, p=0.5),
                                albumentations.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5), max_pixel_value=255.0, always_apply=True, p=1)])


data1 = BasicDataset(transform=transform1)
data2 = jpeg4pyDataset(transform=transform1)
data3 = jpeg4pyalbDataset(transform=transform2)
data4 = turbojpeg4pyalbDataset(transform=transform2)


class data_prefetcher():
    def __init__(self, loader, fp16=False):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1, 3, 1, 1)
        self.std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1, 3, 1, 1)
        # With amp, it isn't necessary to manually convert data to half.
        self.fp16 = fp16
        # if self.fp16:
        #     self.mean = self.mean.half()
        #     self.std = self.std.half()
        self.preload()
        
    def preload(self):
        try:
            # self.next_input, self.next_target = next(self.loader)
            self.next_input = next(self.loader)
        except StopIteration:
            # self.next_input, self.next_target = None, None
            self.next_input = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            # self.next_target = self.next_target.cuda(non_blocking=True)
            # if self.fp16:
            #     self.next_input = self.next_input.half().to(device='cuda:0', non_blocking=True)
            # else:
            self.next_input = self.next_input.to(device='cuda:0', non_blocking=True)
            # self.next_input = self.next_input.sub_(self.mean).div_(self.std)
            
    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.next_input
        self.preload()
        return batch


training_data_loader = DataLoader(dataset=data4, num_workers=8, batch_size=64, pin_memory=True, shuffle=True)

# for iteration, batch in enumerate(training_data_loader, 1):
#     # train code


data_loader = data_prefetcher(training_data_loader)
s = time.time()
data = data_loader.next()
iteration = 0
while data is not None:
    # train code
    iteration += 1
    data = data_loader.next()
print(time.time() - s)
