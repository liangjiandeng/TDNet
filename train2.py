import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F
from data import *
from model import TDNet

import numpy as np
from tensorboardX import SummaryWriter
import shutil


# ================== Pre-Define =================== #


SEED = 10
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
# cudnn.benchmark = True  ###自动寻找最优算法
cudnn.deterministic = True

# ============= HYPER PARAMS(Pre-Defined) ==========#
lr = 0.001
epochs = 300
ckpt = 50
batch_size = 32
model = TDNet().cuda()

model_path = "Weight_1104_new/.pth"


if os.path.isfile(model_path):
    # Load the pretrained Encoder
    model.load_state_dict(torch.load(model_path))
    print('PANnet is Successfully Loaded from %s' % (model_path))
#
# summaries(model, grad=True)



criterion = nn.MSELoss().cuda()


optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9,0.999))   # optimizer 1
#optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-7)
#lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=180, gamma=0.1) # lr = lr* 1/gamma for each step_size = 180

if os.path.exists('train_logs'):  # for tensorboard: copy dir of train_logs
   shutil.rmtree('train_logs')   # ---> console: tensorboard --logdir = dir of train_logs
writer = SummaryWriter('train_logs')

# def save_checkpoint(model, epoch): # save model function
#     model_folder = "Weights_1019_pannew/"
#     model_out_path = model_folder + "{}.pth".format(epoch)
#     state = {"epoch": epoch, "model": model}
#     if not os.path.exists(model_folder):
#         os.makedirs(model_folder)
#
#     torch.save(state, model_out_path)
#     print("Checkpoint saved to {}".format(model_out_path))
def save_checkpoint(model, epoch):  # save model function
    model_out_path = 'Weight_1104_new' + '/' + "{}.pth".format(epoch)
    # if not os.path.exists(model_out_path):
    #     os.makedirs(model_out_path)
    torch.save(model.state_dict(), model_out_path)


###################################################################
# ------------------- Main Train ----------------------------------
###################################################################


def train(training_data_loader, validate_data_loader):
    print('Start training...')

    for epoch in range(epochs):

        epoch += 1
        epoch_train_loss, epoch_val_loss = [], []
        # if (epoch <= 500):
        #   lamda = 1 - (epoch//5)*0.01
        # else:
        #   lamda = 0
        lamda = 0.4
        # ============Epoch Train=============== #
        model.train()
        for iteration, batch in enumerate(training_data_loader, 1):
            gt, lms, ms, pan = Variable(batch[0], requires_grad=False).cuda(), \
                               Variable(batch[1]).cuda(), \
                               Variable(batch[2]).cuda(), \
                               Variable(batch[3]).cuda()
           # gt, lms, ms, pan= Variable(batch[0]), Variable(batch[1]), Variable(batch[2]), Variable(
       # batch)
    ###GPU加速
              # expand to N*H*W*1
            gt_down = F.interpolate(gt, scale_factor=0.4).cuda()

            optimizer.zero_grad() # fixed

            out1,out2 = model(ms.float(), pan.float()) # call model

            loss1 = criterion(out1.float(), gt_down.float())
            loss2 = criterion(out2.float(), gt.float())
            loss = lamda*loss1+(1-lamda)*loss2 # compute loss
            epoch_train_loss.append(loss.item()) # save all losses into a vector for one epoch


            loss.backward() # fixed

            optimizer.step() # fixed

   #     lr_scheduler.step()  # update lr

        t_loss = np.nanmean(np.array(epoch_train_loss)) # compute the mean value of all losses, as one epoch loss
        Loss0 = np.array(epoch_train_loss)
        #np.save('/home/office-401-2/Desktop/Machine Learning/Tian-Jing Zhang/Dataset_ZHANG/BDPN_MRA/loss/epoch_{}'.format(epoch),Loss0)
        #writer.add_scalar('train/loss', t_loss, epoch) # write to tensorboard to check
        print('Epoch: {}/{} training loss: {:.7f}'.format(epochs, epoch, t_loss)) # print loss for each epoch

        if epoch % ckpt == 0:  # if each ckpt epochs, then start to save model
            save_checkpoint(model, epoch)
        # ============Epoch Validate=============== #
        model.eval()
        with torch.no_grad():
            for iteration, batch in enumerate(validate_data_loader, 1):
                gt, lms, ms, pan = Variable(batch[0], requires_grad=False).cuda(), \
                                   Variable(batch[1]).cuda(), \
                                   Variable(batch[2]).cuda(), \
                                   Variable(batch[3]).cuda()
                ###GPU加速
                #pan = pan[:, np.newaxis,:, :].permute # expand to N*H*W*1
                gt_down = F.interpolate(gt, scale_factor=0.5).cuda()

                out1, out2 = model(ms.float(), pan.float())  # call model

                loss1 = criterion(out1.float(), gt_down.float())
                loss2 = criterion(out2.float(), gt.float())
                loss = lamda * loss1 + (1 - lamda) * loss2  # compute loss
                epoch_val_loss.append(loss.item())

        v_loss = np.nanmean(np.array(epoch_val_loss))
        #writer.add_scalar('val/loss', v_loss, epoch)
        print('             validate loss: {:.7f}'.format(v_loss))



###################################################################
# ------------------- Main Function (Run first) -------------------
###################################################################
if __name__ == "__main__":
    train_set = DatasetFromHdf5('./train.h5') # creat data for training
    training_data_loader = DataLoader(dataset=train_set, num_workers=0, batch_size=batch_size, shuffle=True,
                                      pin_memory=True, drop_last=True) # put training data to DataLoader for batches
    #training_data_loader = DataLoader (dataset=MyDataset(traindata_path), num_workers=0, batch_size=batch_size, shuffle=True,
                      #                pin_memory=True, drop_last=True)
    #validation_data_loader = DataLoader(dataset=MyDataset(validationdata_path), num_workers=0, batch_size=batch_size, shuffle=True,
                                   #  pin_memory=True, drop_last=True)
    validate_set = DatasetFromHdf5('./valid.h5') # creat data for validation
    validate_data_loader = DataLoader(dataset=validate_set, num_workers=0, batch_size=batch_size, shuffle=True,
                                      pin_memory=True, drop_last=True) # put training data to DataLoader for batches

    train(training_data_loader, validate_data_loader)  # call train function (call: Line 53)
