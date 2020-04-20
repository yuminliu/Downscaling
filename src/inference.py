#%%
import os
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from skimage.transform import resize
from models import YNet30
from utils import AverageMeter

cudnn.benchmark = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#%%
variable = 'ppt' #'tmin' #'ppt' #'tmax'
resolution = 0.5
scale = 2 # downscaling factor
input_channels = 35 #33 # 18
folder = '2020-01-08_14.44.21.689965_debug'
epoch = 299
#modelname = 'YNet30' #'ESPCN' #
#weightspath = '../results/Climate/PRISM_GCM/REDNet30/scale{}/2019-12-09_16.53.20.651236/REDNet30_epoch_{}.pth'.format(scale,299)
weightspath = '../results/Climate/PRISM_GCM/YNet30/{}/scale{}/{}/YNet30_epoch_{}.pth'.format(variable,scale,folder,epoch)
datapath = '../data/Climate/PRISM_GCMdata/{}/{}by{}/test/'.format(variable,resolution,resolution)
#filename = 'cpc_gcm_log1p_prmean_monthly_0.25to1.0_195001-200512_USA_month637.npz'
filenames = [f for f in os.listdir(datapath) if f.endswith('.npz')]
#filenames = [filenames[0]]
filenames = sorted(filenames)
savepath = '/'.join(weightspath.split('/')[:-1])+'/' #'../results/Climate/CPC_GCM/'
datapath2 = None #'../data/Climate/PRISM_GCMdata/prism_gcm_log1p_prmean_monthly_0.125by0.125_195001-199912_USA_month_1to600.npy'
use_climatology = True
#%%
#if __name__ == '__main__':
model = YNet30(input_channels=input_channels,output_channels=1,scale=scale,use_climatology=use_climatology)

#model = ESPCNNet(upscale_factor=scale)

checkpoint = torch.load(weightspath, map_location=lambda storage, loc: storage)
model.load_state_dict(checkpoint['model_state_dict'])
#state_dict = model.state_dict()
#for n, p in torch.load(weightspath, map_location=lambda storage, loc: storage).items():
#    if n in state_dict.keys():
#        state_dict[n].copy_(p)
#    else:
#        raise KeyError(n)

model = model.to(device)
model.eval()

if savepath and not os.path.exists(savepath):
    os.makedirs(savepath)

total_losses = AverageMeter()
criterion = torch.nn.MSELoss() 
losses = 0 
preds = [] 
for filename in filenames:
    data = np.load(datapath+filename)
    X = data['gcms'] # tmax tmin:[-1,1], ppt: [0,5], [Ngcm,Nlat,Nlon]
    
    X = X/5.0 #[0.0,1.0]
    
    gcms = torch.from_numpy(X).float() #[Ngcm,Nlat,Nlon]
    #gcms = torch.mean(gcms,dim=0,keepdim=True) #[1,Nlat,Nlon]
    #target = torch.from_numpy(data['cpc']).float() #[1,Nlat,Nlon]
    target = torch.from_numpy(data['prism']).float() #tmax tmin:[-1,1], ppt:[0,5], [1,Nlat,Nlon]
    
    #target = target/5.0 #[0.0,1.0]
    
    target = target.to(device)
    
    if use_climatology:
        input1 = gcms.unsqueeze(dim=0) #[1,Ngcm,Nlat,Nlon]
        
        #X2 = data['climatology'] # [1,Nlat,Nlon]
        X21 = data['climatology'] # [0,2], [1,Nlat,Nlon]
        X22 = data['elevation'] # [1,Nlat,Nlon]
        
        
        X21 = X21/5.0 #[0.0,1.0]
        X22 = X22/10.0 #[0.0,1.0]
        
        
        X2 = np.concatenate((X21,X22),axis=0)  # [2,Nlat,Nlon]
        
        X3 = resize(np.transpose(X,axes=(1,2,0)),data['prism'][0,:,:].shape,order=1,preserve_range=True) # [Nlat,Nlon,Ngcm]
        X3 = np.transpose(X3,axes=(2,0,1))# [Ngcm,Nlat,Nlon]
        input3 = torch.from_numpy(X3[np.newaxis,...]).float() # [Ngcm,Nlat,Nlon]
        
        input2 = torch.from_numpy(X2[np.newaxis,...]).float() #[1,1,Nlat,Nlon] --> [1,1,Nlat,Nlon]
        
        #inputs = [input1,input2]  
        inputs = [input1,input2,input3]
        inputs = [e.to(device) for e in inputs]    
        with torch.no_grad():
            pred = model(*inputs) # [1,1,Nlat,Nlon]
    else:
        inputs = gcms.unsqueeze(dim=0) #[1,Ngcm,Nlat,Nlon]
        inputs = inputs.to(device)
        with torch.no_grad():
            pred = model(inputs) # [1,1,Nlat,Nlon]
    #print('output pred.size()={}'.format(pred.size())) # [1,1,Nlat,Nlon]
    
    pred = pred*5.0 #[0.0,5.0]
    #target = target*5.0 #[0.0,5.0]
    
    pred = pred.squeeze_().expm1_() # [Nlat,Nlon] unit: mm/day
    target = target.squeeze_().expm1_() # [Nlat,Nlon] unit: mm/day
    #pred = pred.squeeze_()*50.0 # [Nlat,Nlon] unit: Celsius
    #target = target.squeeze_()*50.0 # [Nlat,Nlon] unit: Celsius
    #pred = (pred.squeeze_()-1.0)*50.0 # [Nlat,Nlon] unit: Celsius
    #target = (target.squeeze_())*50.0 # [Nlat,Nlon] unit: Celsius
    
    loss = criterion(pred, target)
    total_losses.update(loss.item(),1)
    
    losses += np.mean((pred.cpu().numpy()-target.cpu().numpy())**2)
    
    preds.append(pred)

preds = torch.stack(preds,dim=0) #[Ntest,Nlat,Nlon], unit: mm/day
losses /= len(filenames)    
print('downscaling MSE: {}'.format(total_losses.avg))
print('losses MSE: {}'.format(losses))

#%%
target = target.cpu().numpy()
pred = pred.cpu().numpy()
preds = preds.cpu().numpy()

if savepath:
    np.save(savepath+'pred_results_MSE{}.npy'.format(losses),preds)


if use_climatology:
    inputs = inputs[0]
inputs = inputs.cpu().numpy()
inputs = np.mean(np.squeeze(inputs)*50.0,axis=0) # [Nlat,Nlon]
#inputs = np.mean(np.expm1(np.squeeze(inputs)),axis=0) # [Nlat,Nlon]
#inputs = np.expm1(np.squeeze(inputs)) # [Nlat,Nlon]
#%%
verbose = True
if verbose:
    #### plot figures
    import matplotlib.pyplot as plt 
    fig,axs = plt.subplots(2,2)
    #axs[0,0].imshow(target/target.max())
    axs[0,0].imshow(target)
    axs[0,0].set_title('target')
    #axs[1,0].imshow(pred/pred.max())
    axs[1,0].imshow(pred)
    axs[1,0].set_title('pred: predicted result')
    #plt.show()
    #axs[0,1].imshow(inputs/inputs.max())
    axs[0,1].imshow(inputs)
    axs[0,1].set_title('input gcm')
    
    diff_target_pred = np.abs(target-pred)

    #diff1 = axs[1,1].imshow(diff_target_pred/diff_target_pred.max())
    diff1 = axs[1,1].imshow(diff_target_pred)
    axs[1,1].set_title('abs(target-pred)')

    
    fig.colorbar(diff1,ax=axs[1,1],fraction=0.05)

    
    ## hide x labels and  tick labels for top plots and y ticks for right plots
    for ax in axs.flat:
        ax.label_outer()
    if savepath:
        plt.savefig(savepath+'pred_vs_groundtruth.png',dpi=1200,bbox_inches='tight')
    plt.show()
    
    
    # fig, axs = plt.subplots(6,3)
    # for i in range(6):
    #     for j in range(3):
    #         axs[i,j].imshow(gcms[:,:,i*3+j])
    #         axs[i,j].set_title('{}'.format(i*3+j+1))
    # ## hide x labels and  tick labels for top plots and y ticks for right plots
    # for ax in axs.flat:
    #     ax.label_outer()
    #plt.show()

