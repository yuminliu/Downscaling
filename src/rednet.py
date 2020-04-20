#%%
import os
import time
import datetime
import numpy as np
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import SubsetRandomSampler
from dataset import myDataset_REDNet,ImageNetDataset
from models import REDNet30
from utils import AverageMeter, train_one_epoch, validate, create_savepath, save_checkpoint, load_checkpoint

#%%
start_time = time.time()
variable = 'ppt' #'tmin' #'tmax'
resolution = 0.125
train_datapath = '../data/Climate/PRISM_GCMdata/{}/{}by{}/train/'.format(variable,resolution,resolution)
valid_datapath = '../data/Climate/PRISM_GCMdata/{}/{}by{}/val/'.format(variable,resolution,resolution)
save_root_path = '../results/Climate/PRISM_GCM/REDNet30/{}/'.format(variable)
datapath2 = None #'../data/Climate/PRISM_GCMdata/prism_gcm_log1p_prmean_monthly_0.125by0.125_195001-199912_USA_month_1to600.npy'
#save_root_path = '../results/Climate/PRISM_GCM/ESPCN/'
##prefix = '/home/yumin/myProgramFiles/DATA/ImageNet/'
#prefix = '/home/yumin/Desktop/DS/DATA/ImageNet/'
##prefix = '/scratch/wang.zife/YuminLiu/DATA/'
#train_datapath = prefix+'ImageNetData_256by256_JPEG/train/'
#valid_datapath = prefix+'ImageNetData_256by256_JPEG/val/'
#save_root_path = '../results/ImageNet/'

is_debug = False #True
is_resuming = False #True
is_transfer = False #True
nSubSample = 500 # number of samples to select
num_epochs = 300#500#100#5
batch_size = 32 #64 #128
lr = 1e-4
lr_patience = 10
num_workers = 8
patch_size = (256,256) #(64,64)
epoch_start = 0
scale = 8 # downscale factor
input_channels = 1
seed = 123
#use_climatology = True
model_name = 'REDNet30' #'ESPCN' #
save_root_path += 'scale{}/'.format(scale)
torch.manual_seed(seed)
cudnn.benchmark = True # true if input size not vary else false
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('using device {}'.format(device))
nGPU = torch.cuda.device_count() # number of GPU used, 0,1,..

#%%
train_dataset = myDataset_REDNet(datapath=train_datapath,datapath2=datapath2)
valid_dataset = myDataset_REDNet(datapath=valid_datapath,datapath2=datapath2)

#train_dataset = ImageNetDataset(datapath=train_datapath,is_train=True,scale=scale,patch_size=patch_size)
#if is_debug:
#    train_dataset = torch.utils.data.Subset(train_dataset,indices=list(range(0,nSubSample)))
#valid_dataset = ImageNetDataset(datapath=valid_datapath,is_train=False,scale=scale,patch_size=patch_size)
#if is_debug:
#    valid_dataset = torch.utils.data.Subset(valid_dataset,indices=list(range(0,nSubSample)))

print('len(train_dataset)={}'.format(len(train_dataset)))
print('len(valid_dataset)={}'.format(len(valid_dataset)))
    
train_loader = DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers)
valid_loader = DataLoader(dataset=valid_dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers)


#%%
model = REDNet30(input_channels=input_channels,output_channels=1)

#model = ESPCNNet(upscale_factor=scale)

#print('model=\n{}'.format(model))
if nGPU>1:
    print('Using {} GPUs'.format(nGPU))
    model = torch.nn.DataParallel(model)
    
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
train_losses, valid_losses = [], []

#is_resuming = True
#if is_resuming:
#    #checkpoint_path = '../results/ImageNet/saved/2019-11-14_21.43.07.504184/'
#    #checkpoint_name = 'REDNet30_epoch_21.pth'
#    #checkpoint = torch.load(checkpoint_path+checkpoint_name)
#    
#    checkpoint_path = '../results/ImageNet/2019-11-21_17.22.46.173887_debug/'
#    checkpoint_name = 'REDNet30_epoch_3.pth'
#    checkpoint = torch.load(checkpoint_path+checkpoint_name,map_location=device)
#    
#    model.load_state_dict(checkpoint['model_state_dict'])
#    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#    epoch_start = checkpoint['epoch']+1
#    train_losses = checkpoint['train_losses']
#    valid_losses = checkpoint['valid_losses']
#    #model.train()
#    
#    print('resume training')


if is_resuming:
    #checkpoint_path = '../results/ImageNet/saved/2019-11-14_21.43.07.504184/'
    checkpoint_path = '../results/Climate/PRISM_GCM/REDNet30/scale2//'
    checkpoint_name = 'REDNet30_epoch_99.pth'
    load_res = load_checkpoint(checkpoint_path,checkpoint_name,model,optimizer,device,nGPU)
    model = load_res['model']
    optimizer = load_res['optimizer']
    epoch_start = load_res['epoch_start']
    train_losses = load_res['train_losses']
    valid_losses = load_res['valid_losses']
    print('resuming training')
else:
    checkpoint_path = None




#model = model.to(device)
criterion = torch.nn.MSELoss()
lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',patience=lr_patience)
#train_losses, valid_losses = [], []
#%% create save path 
savepath = checkpoint_path if (not is_transfer and is_resuming) else create_savepath(rootpath=save_root_path,is_debug=is_debug)
#if is_resuming:
#    savepath = checkpoint_path
#else:
#    savepath = create_savepath(savepath=save_root_path,is_debug=is_debug)
for epoch in range(epoch_start,epoch_start+num_epochs):
    train_loss = train_one_epoch(model,optimizer,criterion,train_loader,epoch,device,epoch_start+num_epochs)
    train_losses.append(train_loss)
    valid_loss = validate(model,criterion,valid_loader,device)
    valid_losses.append(valid_loss)
    
    lr_scheduler.step(valid_loss)
    #print('epoch {} done!'.format(epoch))


    save_checkpoint(savepath=savepath,epoch=epoch,model=model,optimizer=optimizer,
    train_losses=train_losses,valid_losses=valid_losses,
    lr=lr,lr_patience=lr_patience,model_name=model_name,nGPU=nGPU)



#%%
#import torch
##data = torch.load('../results/ImageNet/saved/2019-11-14_21.43.07.504184/REDNet30_epoch_40.pth',map_location=torch.device('cpu'))
#data = torch.load('../results/Climate/PRISM_GCM/REDNet30/scale2/2019-12-04_10.33.30.784658/REDNet30_epoch_180.pth',map_location=torch.device('cpu'))
#train_losses, valid_losses = data['train_losses'], data['valid_losses']

# plot losses
verbose = True # False
if verbose:
    import matplotlib.pyplot as plt 
    xx = range(1,len(train_losses)+1)
    fig = plt.figure()
    plt.plot(xx,train_losses,'b--',label='train_losses')
    plt.plot(xx,valid_losses,'r-',label='valid_losses')
    plt.legend()
    plt.title('losses')
    plt.xlabel('epoch')
    plt.ylabel('loss(avgerage MSE)')
    savename = 'losses'
    #savepath = '../results/ImageNet/saved/2019-11-14_21.43.07.504184/'
    if savepath:
        plt.savefig(savepath+savename+'.png',dpi=1200,bbox_inches='tight')
    plt.show()

end_time = time.time()
print('Job done! total time: {} seconds!'.format(end_time-start_time))

torch.cuda.empty_cache()