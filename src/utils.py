class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


import torch
from tqdm import tqdm
def train_one_epoch(model,optimizer,criterion,train_loader,epoch,device,num_epochs):
    model.train()
    total_losses = AverageMeter()
    with tqdm(total=len(train_loader)) as _tqdm:
        _tqdm.set_description('train epoch: {}/{}'.format(epoch + 1, num_epochs))
        for data in train_loader:
            inputs, labels = data
            labels = labels.to(device)
            if isinstance(inputs,list): # multiple inputs
                #print('len(inputs)={},inputs[0].size()={},inputs[1].size()={},labels.size()={}'.format(len(inputs),inputs[0].size(),inputs[1].size(),labels.size()))
                inputs = [e.to(device) for e in inputs]
                preds = model(*inputs)
            else: # single input
                #print('inputs.size()={},labels.size()={}'.format(inputs.size(),labels.size()))
                inputs = inputs.to(device)
                preds = model(inputs)
            ##print('inputs.size()={},labels.size()={}'.format(inputs.size(),labels.size()))
            #inputs, labels = inputs.to(device), labels.to(device)
            #preds = model(inputs)
            loss = criterion(preds, labels)
            if isinstance(inputs,list):
                total_losses.update(loss.item(), len(inputs[0]))
            else:
                total_losses.update(loss.item(), len(inputs))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            _tqdm.set_postfix(loss='{:.6f}'.format(total_losses.avg))
            _tqdm.update(1)
    return total_losses.avg

@torch.no_grad()
def validate(model,criterion,valid_loader,device):
    model.eval()
    total_losses = AverageMeter()
    with tqdm(total=len(valid_loader)) as _tqdm:
        _tqdm.set_description('valid progress: ')
        for data in valid_loader:
            inputs, labels = data
            labels = labels.to(device)
            
            if isinstance(inputs,list):
                inputs = [e.to(device) for e in inputs] 
                preds = model(*inputs)
            else:
                inputs = inputs.to(device)
                preds = model(inputs)
            
            loss = criterion(preds, labels)
            if isinstance(inputs,list):
                total_losses.update(loss.item(), len(inputs[0]))
            else:
                total_losses.update(loss.item(), len(inputs))
            _tqdm.set_postfix(loss='{:.6f}'.format(total_losses.avg))
            _tqdm.update(1)
    return total_losses.avg


import os
import datetime
def create_savepath(rootpath='../results/',is_debug=True):
    timestamp = str(datetime.datetime.now()).replace(' ','_').replace(':','.')
    if is_debug:
        savepath = rootpath+timestamp+'_debug/'
    else:
        savepath = rootpath+timestamp+'/'
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    return savepath


#import torch
def save_checkpoint(savepath,epoch,model,optimizer,train_losses,valid_losses,lr,lr_patience,model_name='YNet30',nGPU=0):
    #### save models
    if nGPU>1:
        model_state_dict = model.module.state_dict() # multi GPU version
    else:
        model_state_dict = model.state_dict() # single GPU / CPU version
        
    torch.save({'epoch':epoch,
                'model_state_dict':model_state_dict,
                'optimizer_state_dict':optimizer.state_dict(),
                'train_losses':train_losses,
                'valid_losses':valid_losses,
                'lr':lr,
                'lr_patience':lr_patience}, 
                os.path.join(savepath, '{}_epoch_{}.pth'.format(model_name,epoch)))
    #torch.save({'train_losses':train_losses,'valid_losses':valid_losses},savepath+'losses_epoch_{}.pkl'.format(epoch))

import collections
def load_checkpoint(checkpoint_path,checkpoint_name,model,optimizer,device,nGPU):
    #checkpoint_path = '../results/ImageNet/saved/2019-11-14_21.43.07.504184/'
    #checkpoint_name = 'YNet30_epoch_21.pth'
    #checkpoint = torch.load(checkpoint_path+checkpoint_name)
    
    #checkpoint_path = '../results/ImageNet/2019-11-21_17.22.46.173887_debug/'
    #checkpoint_name = 'YNet30_epoch_3.pth'
    checkpoint = torch.load(checkpoint_path+checkpoint_name,map_location=device)
    model_state_dict = collections.OrderedDict()
    if nGPU>1: # multi GPU version
        for key in checkpoint['model_state_dict']:
            model_state_dict['module.'+key] = checkpoint['model_state_dict'][key]
    else: # single GPU / CPU version
        model_state_dict = checkpoint['model_state_dict']
                   
    model.load_state_dict(model_state_dict)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch_start = checkpoint['epoch']+1
    train_losses = checkpoint['train_losses']
    valid_losses = checkpoint['valid_losses']
    
    res = {'model':model,'optimizer':optimizer,'epoch_start':epoch_start,
           'train_losses':train_losses,'valid_losses':valid_losses}
    
    return res