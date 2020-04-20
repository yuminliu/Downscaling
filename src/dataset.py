import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import random
import glob
import io
import numpy as np
#from PIL import Image
from torch.utils.data import Dataset
#from torchvision.transforms import ToTensor
import torch
from skimage.transform import resize

class myDataset(Dataset):
    def __init__(self, datapath, patch_size=None, transform=None, datapath2=None):
        self.datapaths = sorted(glob.glob(datapath + '*.npz'))       
        #self.patch_size = patch_size
        self.transform = transform
        self.datapath2 = datapath2


    def __getitem__(self, idx):
        data = np.load(self.datapaths[idx])

        X = torch.from_numpy(data['gcms']).float() #[Ngcm,Nlat,Nlon]
        #X = torch.from_numpy(np.mean(data['gcms'],axis=0,keepdims=True)).float() #[Ngcm,Nlat,Nlon] --> [1,Nlat,Nlon]
        
        X = X/5.0 # -->[0.0,1.0]
        
        
        X21 = torch.from_numpy(data['climatology']).float() #[1,Nlat,Nlon]
        X22 = torch.from_numpy(data['elevation']).float() #[1,Nlat,Nlon]
        
        X21 = X21/5.0 #[0.0,1.0]
        X22 = X22/10.0 #[0.0,1.0]
        
        X2 = torch.cat([X21,X22],dim=0) # [2,Nlat,Nlon]
        
        X30 = data['gcms'] # [Ngcm,Nlat,Nlon]
        
        X30 = X30/5.0 # [0.0,1.0]
        
        X30 = resize(np.transpose(X30,axes=(1,2,0)),data['prism'][0,:,:].shape,order=1,preserve_range=True) # [Nlat,Nlon,Ngcm]
        X30 = np.transpose(X30,axes=(2,0,1))# [Ngcm,Nlat,Nlon]
        X3 = torch.from_numpy(X30).float() # [Ngcm,Nlat,Nlon]
        
        
        
        y = torch.from_numpy(data['prism']).float() #[1,Nlat,Nlon]
        
        y = y/5.0 #[0.0,1.0]
        
        #print('X.size()[-2:]={}'.format(X.size()[-2:]))
        #print('y.size()={}'.format(y.size()))
        # if len(y.size())<len(X.size()):
        #     #assert y.size()==X.size()[-2:], 'X and y should have the same height and width!'
        #     y = torch.unsqueeze(y,dim=0) # [1,Nlat,Nlon]
        
#        y2 = data['prism']#[1,Nlat,Nlon]
#        y = torch.from_numpy(y2).float() #[1,Nlat,Nlon]
#        y3 = np.squeeze(y2)#[1,Nlat,Nlon] --> [Nlat,Nlon]
#        
#        X2 = np.mean(data['gcms'],axis=0) #[Ngcm,Nlat,Nlon] --> [Nlat,Nlon] 
#        X = resize(X2,output_shape=y3.shape,order=1,preserve_range=True) #[Nlat,Nlon] 
#        X = torch.from_numpy(X[np.newaxis,...]).float() #[Nlat,Nlon] --> [1,Nlat,Nlon]
#        #print('X.size()={},y.size()={}'.format(X.size(),y.size()))
    
        if self.transform:
            y = self.transform(y)
            X = self.transform(X)

#        if self.datapath2:
#            X2 = np.load(self.datapath2) #[12,Nlat,Nlon]
#            X2 = torch.from_numpy(X2).float()
#            X = [X,X2]
        X = [X,X2,X3]

        return X, y

    def __len__(self):
        return len(self.datapaths)


class myDataset_REDNet(Dataset):
    def __init__(self, datapath, patch_size=None, transform=None, datapath2=None):
        self.datapaths = sorted(glob.glob(datapath + '*.npz'))       
        #self.patch_size = patch_size
        self.transform = transform
        self.datapath2 = datapath2


    def __getitem__(self, idx):
        data = np.load(self.datapaths[idx])
        y = np.squeeze(data['prism']) # [1,Nlat,Nlon] --> [Nlat,Nlon]
        gcms = np.mean(data['gcms'],axis=0)# [Ngcm,Nlat,Nlon] --> [Nlat,Nlon]
        
        y = y/5.0 # ppt: [0.0,1.0]
        gcms = gcms/5.0 # ppt: [0.0,1.0]
        
        #y = np.squeeze(data['prism'])+1.0 # tmax,tmin: [-1,1]-->[0,2], [1,Nlat,Nlon] --> [Nlat,Nlon]
        #gcms = np.mean(data['gcms'],axis=0)+1.0 # tmax,tmin: [-1,1]-->[0,2], [Ngcm,Nlat,Nlon] --> [Nlat,Nlon]
        
        gcms = resize(gcms,y.shape,order=1,preserve_range=True) #[Nlat,Nlon]
        X = torch.from_numpy(gcms[np.newaxis,...]).float() #[Nlat,Nlon] --> [1,Nlat,Nlon]
        #gcms = np.transpose(data['gcms'],axes=(1,2,0)) #[Ngcm,Nlat,Nlon] --> [Nlat,Nlon,Ngcm]
        #gcms = resize(gcms,output_shape=y.shape,order=1,preserve_range=True)
        #gcms = np.transpose(gcms,axes=(2,0,1)) #[Nlat,Nlon,Ngcm] --> [Ngcm,Nlat,Nlon]

        #X = torch.from_numpy(gcms).float() #[Ngcm,Nlat,Nlon]
        #X = torch.from_numpy(np.mean(gcms,axis=0,keepdims=True)).float() #[Ngcm,Nlat,Nlon] --> [1,Nlat,Nlon]
        
        #X21 = torch.from_numpy(data['climatology']).float() #[1,Nlat,Nlon]
        #X22 = torch.from_numpy(data['elevation']).float() #[1,Nlat,Nlon]
        #X2 = torch.cat([X21,X22],dim=0) # [2,Nlat,Nlon]
        #X = torch.cat([X,X21,X22],dim=0) # [20,Nlat,Nlon]
        y = torch.from_numpy(y[np.newaxis,...]).float() #[1,Nlat,Nlon]
        
        
        
        #print('X.size()[-2:]={}'.format(X.size()[-2:]))
        #print('y.size()={}'.format(y.size()))
        # if len(y.size())<len(X.size()):
        #     #assert y.size()==X.size()[-2:], 'X and y should have the same height and width!'
        #     y = torch.unsqueeze(y,dim=0) # [1,Nlat,Nlon]
        
#        y2 = data['prism']#[1,Nlat,Nlon]
#        y = torch.from_numpy(y2).float() #[1,Nlat,Nlon]
#        y3 = np.squeeze(y2)#[1,Nlat,Nlon] --> [Nlat,Nlon]
#        
#        X2 = np.mean(data['gcms'],axis=0) #[Ngcm,Nlat,Nlon] --> [Nlat,Nlon] 
#        X = resize(X2,output_shape=y3.shape,order=1,preserve_range=True) #[Nlat,Nlon] 
#        X = torch.from_numpy(X[np.newaxis,...]).float() #[Nlat,Nlon] --> [1,Nlat,Nlon]
#        #print('X.size()={},y.size()={}'.format(X.size(),y.size()))
    
        if self.transform:
            y = self.transform(y)
            X = self.transform(X)

#        if self.datapath2:
#            X2 = np.load(self.datapath2) #[12,Nlat,Nlon]
#            X2 = torch.from_numpy(X2).float()
#            X = [X,X2]
        #X = [X,X2]

        return X, y

    def __len__(self):
        return len(self.datapaths)


class myDataset_ESPCN(Dataset):
    def __init__(self, datapath, patch_size=None, transform=None, datapath2=None):
        self.datapaths = sorted(glob.glob(datapath + '*.npz'))       
        #self.patch_size = patch_size
        self.transform = transform
        self.datapath2 = datapath2


    def __getitem__(self, idx):
        data = np.load(self.datapaths[idx])
        #X = data['gcms']
        X = (data['gcms']+1.0)/2.0 # tmax/tmin: [-1,1]-->[0,1]
        
        #X = data['gcms']/5.0 # ppt: [0.0,1.0]

        X = torch.from_numpy(X).float() #[Ngcm,Nlat,Nlon]
        #X = torch.from_numpy(np.mean(data['gcms'],axis=0,keepdims=True)).float() #[Ngcm,Nlat,Nlon] --> [1,Nlat,Nlon]
        
        #X21 = torch.from_numpy(data['climatology']).float() #[1,Nlat,Nlon]
        #X22 = torch.from_numpy(data['elevation']).float() #[1,Nlat,Nlon]
        #X2 = torch.cat([X21,X22],dim=0) # [2,Nlat,Nlon]
        
        #y = data['prism']
        y = (data['prism']+1.0)/2.0 # tmax/tmin: [-1,1]-->[0,1]
        
        #y = data['prism']/5.0 # ppt: [0.0,1.0]
        
        y = torch.from_numpy(y).float() #[1,Nlat,Nlon]

    
        if self.transform:
            y = self.transform(y)
            X = self.transform(X)

        return X, y

    def __len__(self):
        return len(self.datapaths)





from skimage.io import imread
#from skimage.transform import resize
from skimage.color import rgb2gray
#import numpy as np
class ImageNetDataset(Dataset):
    def __init__(self, datapath, is_train=True, scale=4, patch_size=None, transform=None):
        if is_train:
            self.datapaths = sorted(glob.glob(datapath + '*/*.JPEG'))
        else:
            self.datapaths = sorted(glob.glob(datapath + '*/*.JPEG'))
        self.scale = scale
        self.patch_size = patch_size
        self.transform = transform

    def __getitem__(self, idx):
        img = imread(self.datapaths[idx])/255.0
        #assert len(img.shape)==3, "error: img.shape={}!=3\n".format(img.shape)
        #if len(img.shape)==2:
        #    img = np.repeat(img[:,:,np.newaxis],repeats=3,axis=2)
        #    img += 0.1*np.random.random(size=img.shape)
        
        if self.patch_size:
            crop_I = np.random.randint(0,img.shape[0]-self.patch_size[0]+1)
            crop_J = np.random.randint(0,img.shape[1]-self.patch_size[1]+1)
            img = img[crop_I:crop_I+self.patch_size[0],crop_J:crop_J+self.patch_size[1],:]

        H,W,C = img.shape
        #H, W = H-H%self.scale, W-W%self.scale
        #img = img[0:H,0:W,:]
        #print('img.shape={}'.format(img.shape))
        X = []
        for order in range(6):
            #temp = resize(img,output_shape=(W//scale,H//scale,C),order=order,preserve_range=True)
            X.append(resize(img,output_shape=(H//self.scale,W//self.scale),order=order,preserve_range=True))
        X = np.transpose(np.concatenate(X,axis=2),axes=(2,0,1)) # [C,H,W]
        y = rgb2gray(img) #[H,W]
        y = y[np.newaxis,:,:] #[1,H,W]

        X = torch.from_numpy(X).float() #[C,H,W]
        y = torch.from_numpy(y).float() #[1,H,W]
        #print('X.size()={}'.format(X.size()))
        #print('y.size()={}'.format(y.size()))
        # if len(y.size())<len(X.size()):
        #     #assert y.size()==X.size()[-2:], 'X and y should have the same height and width!'
        #     y = torch.unsqueeze(y,dim=0) # [1,Nlat,Nlon]
        if self.transform:
            y = self.transform(y)
            X = self.transform(X)

        return X, y

    def __len__(self):
        return len(self.datapaths)




class PRISMDataset(Dataset):
    def __init__(self, datapath, use_gcm=True, patch_size=None, transform=None):
        self.datapaths = sorted(glob.glob(datapath + '*.npz'))   
        self.use_gcm = use_gcm
        self.patch_size = patch_size
        self.transform = transform

    def __getitem__(self, idx):
        data = np.load(self.datapaths[idx])
        y = data['prism'] # [1,Nlat,Nlon]        
        elevation = data['elevation'] #[1,Nlat,Nlon]

        #%% interpolate to the same size with y
        if self.use_gcm:
            gcms = np.mean(data['gcms'],axis=0) #[Ngcm,Nlat,Nlon] --> [Nlat,Nlon]
                        
        else:
            gcms = resize(y[0,:,:],(y.shape[1]//2,y.shape[2]//2),order=1,preserve_range=True) #

        #print('before interpolation: gcms.shape={}'.format(gcms.shape))
        gcms = resize(gcms,y.shape[1:],order=1,preserve_range=True) #
        #print('after interpolation: gcms.shape={}'.format(gcms.shape))  
        
        X = np.concatenate((gcms[np.newaxis,...],elevation),axis=0) # [C,Nlat,Nlon]
        
        if self.patch_size:
            crop_I = np.random.randint(0,X.shape[1]-self.patch_size[0]+1)
            crop_J = np.random.randint(0,X.shape[2]-self.patch_size[1]+1)
            X = X[:,crop_I:crop_I+self.patch_size[0],crop_J:crop_J+self.patch_size[1]]
            y = y[:,crop_I:crop_I+self.patch_size[0],crop_J:crop_J+self.patch_size[1]]
            
        X = torch.from_numpy(X).float() #[Nlat,Nlon,2]      
        y = torch.from_numpy(y).float() #[1,Nlat,Nlon]

        if self.transform:
            y = self.transform(y)
            X = self.transform(X)

        return X, y

    def __len__(self):
        return len(self.datapaths)


class PRISMDataset_DeepSD2(Dataset):
    def __init__(self, datapath, use_gcm=True, patch_size=None, transform=None,base_scale_factor=2):
        self.datapaths = sorted(glob.glob(datapath + '*.npz'))   
        self.use_gcm = use_gcm
        self.patch_size = patch_size
        self.transform = transform
        self.bsf = base_scale_factor

    def __getitem__(self, idx):
        data = np.load(self.datapaths[idx])
        y = data['prism'] # [1,Nlat,Nlon]        
        elevation = np.squeeze(data['elevation']) #[1,Nlat,Nlon] --> [Nlat,Nlon]
        #X21 = torch.from_numpy(data['climatology']).float() #[1,Nlat,Nlon]
        #X21 = X21/5.0 # [0.0,1.0]

        y = y/5.0 # [0.0,1.0]
        elevation = elevation/10.0 # [0.0,1.0]



        #%% interpolate to the same size with y
        if self.use_gcm:
            gcms = np.mean(data['gcms'],axis=0) #[Ngcm,Nlat,Nlon] --> [Nlat,Nlon]     

            gcms = gcms/5.0 # [0.0,1.0]
                   
#        else:
#            gcms = resize(y[0,:,:],(y.shape[1]//2,y.shape[2]//2),order=1,preserve_range=True) #

        #print('before interpolation: gcms.shape={}'.format(gcms.shape))
        elevation1 = resize(elevation,(self.bsf*gcms.shape[0],self.bsf*gcms.shape[1]),order=1,preserve_range=True) #
        elevation2 = resize(elevation,(self.bsf*self.bsf*gcms.shape[0],self.bsf*self.bsf*gcms.shape[1]),order=1,preserve_range=True) 
        #print('after interpolation: gcms.shape={}'.format(gcms.shape))  
        
        #X = np.concatenate((gcms[np.newaxis,...],elevation),axis=0) # [C,Nlat,Nlon]
#        if self.patch_size:
#            crop_I = np.random.randint(0,X.shape[1]-self.patch_size[0]+1)
#            crop_J = np.random.randint(0,X.shape[2]-self.patch_size[1]+1)
#            X = X[:,crop_I:crop_I+self.patch_size[0],crop_J:crop_J+self.patch_size[1]]
#            y = y[:,crop_I:crop_I+self.patch_size[0],crop_J:crop_J+self.patch_size[1]]
         
        y = torch.from_numpy(y).float() #[1,Nlat,Nlon]
        X = torch.from_numpy(gcms[np.newaxis,...]).float() #[1,Nlat,Nlon]      
        X2 = torch.from_numpy(elevation1[np.newaxis,...]).float() #[1,Nlat,Nlon]      
        X3 = torch.from_numpy(elevation2[np.newaxis,...]).float() #[1,Nlat,Nlon]
        X4 = torch.from_numpy(elevation[np.newaxis,...]).float() #[1,Nlat,Nlon]
        #print('y.size()={},X.size()={},X2.size()={},X3.size()={},X4.size()={}'.format(y.size(),X.size(),X2.size(),X3.size(),X4.size()))
        #X = [X,X4]
        #X = [X,X2,X4]
        X = [X,X2,X3,X4]
#        if self.transform:
#            y = self.transform(y)
#            X = self.transform(X)

        return X, y

    def __len__(self):
        return len(self.datapaths)
    
#%%
if __name__=='__main__':
    from torch.utils.data.dataloader import DataLoader
    import matplotlib.pyplot as plt 
    
    datapath = '../data/Climate/PRISM_GCMdata/0.5by0.5/train/'
    dataset = PRISMDataset(datapath,transform=None)
    dataloader = DataLoader(dataset=dataset,batch_size=4)
    dataiter = iter(dataloader)
    X,y = dataiter.next()
    
    print('X.size()={}'.format(X.size()))
    print('y.size()={}'.format(y.size()))
    
    fig = plt.figure()
    plt.imshow(y[0,0,:,:])
    plt.show()

    fig = plt.figure()
    plt.imshow(X[0,0,:,:])
    plt.show()

    fig = plt.figure()
    plt.imshow(X[0,1,:,:])
    plt.show()






















'''
class myDataset(Dataset):
    def __init__(self, images_dir, patch_size, scale, transform=None):
        self.image_files = sorted(glob.glob(images_dir + '*'))
        self.patch_size = patch_size
        self.scale = scale
        self.transform = transform


    def __getitem__(self, idx):
        image = Image.open(self.image_files[idx]).convert('RGB')
        
        # crop to be divisible by scale
        w,h = image.size[0:2]
        w,h = (w//self.scale)*self.scale, (h//self.scale)*self.scale
        image = image.crop((0,0,w,h))
        
        y = image
        X = image.resize((w//self.scale,h//self.scale),resample=Image.LANCZOS) # interpolated to smaller size
        X = image.resize((w,h),resample=Image.LANCZOS) # interpolated to original size, lower resolution
        if self.transform:
            y = self.transform(y)
            X = self.transform(X)
        assert(X!=y,'Error: X==y\n')

        return X, y

    def __len__(self):
        return len(self.image_files)
'''
