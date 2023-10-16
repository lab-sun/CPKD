import os, argparse, time, datetime, stat, shutil
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from util.MF_dataset import MF_dataset
from util.augmentation import RandomFlip, RandomCrop, RandomCropOut, RandomBrightness, RandomNoise
from util.util import compute_results
from sklearn.metrics import confusion_matrix
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from model import Teacher,Student

#############################################################################################
parser = argparse.ArgumentParser(description='Train with pytorch')
############################################################################################# 

parser.add_argument('--Tmodel_name', '-tm', type=str, default='Teacher') 
parser.add_argument('--Smodel_name', '-sm', type=str, default='Student') 
parser.add_argument('--Teacher_model', '-bw', type=str, default='./weights_backup/Teacher/Teacher.pth') 
parser.add_argument('--batch_size', '-b', type=int, default=2) 
parser.add_argument('--gpu', '-g', type=int, default=0)
parser.add_argument('--need_m', '-need_m', type=int, default=7)
parser.add_argument('--sleep', '-sleep', type=int, default=1)
parser.add_argument('--lr_start', '-ls', type=float, default=0.01)
#############################################################################################
parser.add_argument('--lr_decay', '-ld', type=float, default=0.95)
parser.add_argument('--epoch_max', '-em', type=int, default=200) # please stop training mannully 
parser.add_argument('--epoch_from', '-ef', type=int, default=0) 
parser.add_argument('--num_workers', '-j', type=int, default=8)
parser.add_argument('--n_class', '-nc', type=int, default=2)
parser.add_argument('--data_dir', '-dr', type=str, default='./dataset/')
args = parser.parse_args()
#############################################################################################



augmentation_methods = [
    RandomFlip(prob=0.5),
    RandomCrop(crop_rate=0.1, prob=1.0),
    # RandomCropOut(crop_rate=0.2, prob=1.0),
    # RandomBrightness(bright_range=0.15, prob=0.9),
    # RandomNoise(noise_range=5, prob=0.9),
]

def fusion_loss(T_out,S_out):

    T_out_size_B, T_out_size_C, T_out_size_W, T_out_size_H = T_out.size()
    S_out_size_B, S_out_size_C, S_out_size_W, S_out_size_H = S_out.size()
    T_out_P = torch.mean(T_out,dim=1)
    T_out_P = T_out_P.unsqueeze(1)
    T_bn = nn.BatchNorm2d(1).cuda(0)
    T_out_P = T_bn(T_out_P)
    S_out_P = torch.mean(S_out,dim=1)
    S_out_P = S_out_P.unsqueeze(1)
    S_bn = nn.BatchNorm2d(1).cuda(0)
    S_out_P = S_bn(S_out_P)
    S_out_P = F.interpolate(S_out_P,[T_out_size_W,T_out_size_H])
    S_out_P = S_out_P.squeeze(1)
    loss_P = F.mse_loss(T_out_P,S_out_P)

    T_out_C = torch.mean(T_out,dim=2)
    T_out_C = torch.mean(T_out_C,dim=2)
    S_out_C = torch.mean(S_out,dim=2)
    S_out_C = torch.mean(S_out_C,dim=2)
    S_out_C = S_out_C.unsqueeze(1)
    S_out_C = F.interpolate(S_out_C,[T_out_size_C])
    S_out_C = S_out_C.squeeze(1)
    loss_C = F.mse_loss(T_out_C,S_out_C)
    return loss_P+loss_C





def train(epo, Tmodel, Smodel, train_loader, optimizer):
    Smodel.train()
    Tmodel.eval()
    for it, (images, labels, names) in enumerate(train_loader):
        images = Variable(images).cuda(args.gpu)
        labels = Variable(labels).cuda(args.gpu)
        with torch.no_grad():
            T_out1,T_out2,T_out3,T_out4,T_out5,Teacher_logits = Tmodel(images)
            T_out1.detach()
            T_out2.detach()
            T_out3.detach()
            T_out4.detach()
            T_out5.detach()
            Teacher_logits.detach()
        Teacher_label = Teacher_logits.argmax(1)
        start_t = time.time() # time.time() returns the current time
        optimizer.zero_grad()
        S_out1,S_out2,S_out3,S_out4,S_out5,Student_logits = Smodel(images)
        
        loss1 = 0.5*fusion_loss(T_out=T_out1,S_out=S_out1)
        loss2 = 0.5*fusion_loss(T_out=T_out2,S_out=S_out2)
        loss3 = 0.5*fusion_loss(T_out=T_out3,S_out=S_out3)
        loss4 = 0.5*fusion_loss(T_out=T_out4,S_out=S_out4)
        loss5 = 0.5*fusion_loss(T_out=T_out5,S_out=S_out5)
        
        loss_seg = F.cross_entropy(Student_logits, Teacher_label)  # Note that the cross_entropy function has already include the softmax function
        loss = loss_seg+loss1+loss2+loss3+loss4+loss5
        loss.backward()
        optimizer.step()
        lr_this_epo=0
        for param_group in optimizer.param_groups:
            lr_this_epo = param_group['lr']
        print('Train: %s, epo %s/%s, iter %s/%s, lr %.8f, %.2f img/sec, loss %.4f, loss1 %.4f, loss2 %.4f, loss3 %.4f, loss4 %.4f, loss5 %.4f, loss_seg %.4f, time %s' \
            % (args.Smodel_name, epo, args.epoch_max, it+1, len(train_loader), lr_this_epo, len(names)/(time.time()-start_t), float(loss),float(loss1),float(loss2),float(loss3),float(loss4),float(loss5),float(loss_seg),
              datetime.datetime.now().replace(microsecond=0)-start_datetime))
        if accIter['train'] % 1 == 0:
            writer.add_scalar('Train/loss', loss, accIter['train'])
        view_figure = True # note that I have not colorized the GT and predictions here
        if accIter['train'] % 50 == 0:
            if view_figure:
                input_rgb_images = vutils.make_grid(images[:,:3], nrow=8, padding=10) # can only display 3-channel images, so images[:,:3]
                writer.add_image('Train/input_rgb_images', input_rgb_images, accIter['train'])
                scale = max(1, 255//args.n_class) # label (0,1,2..) is invisable, multiply a constant for visualization
                groundtruth_tensor = labels.unsqueeze(1) * scale  # mini_batch*480*640 -> mini_batch*1*480*640
                groundtruth_tensor = torch.cat((groundtruth_tensor, groundtruth_tensor, groundtruth_tensor), 1)  # change to 3-channel for visualization
                groudtruth_images = vutils.make_grid(groundtruth_tensor, nrow=8, padding=10)
                writer.add_image('Train/groudtruth_images', groudtruth_images, accIter['train'])
                predicted_tensor = Student_logits.argmax(1).unsqueeze(1) * scale # mini_batch*args.n_class*480*640 -> mini_batch*480*640 -> mini_batch*1*480*640
                predicted_tensor = torch.cat((predicted_tensor, predicted_tensor, predicted_tensor),1) # change to 3-channel for visualization, mini_batch*1*480*640
                predicted_images = vutils.make_grid(predicted_tensor, nrow=8, padding=10)
                writer.add_image('Train/predicted_images', predicted_images, accIter['train'])
                depth_tensor = images[:,3:4] # mini_batch*args.n_class*480*640 -> mini_batch*480*640 -> mini_batch*1*480*640
                depth_tensor = torch.cat((depth_tensor, depth_tensor, depth_tensor),1) # change to 3-channel for visualization, mini_batch*1*480*640
                depth_tensor = vutils.make_grid(depth_tensor, nrow=8, padding=10)
                writer.add_image('Train/depth_tensor', depth_tensor, accIter['train'])
        accIter['train'] = accIter['train'] + 1

def validation(epo, model, val_loader): 
    model.eval()
    with torch.no_grad():
        for it, (images, labels, names) in enumerate(val_loader):
            images = Variable(images).cuda(args.gpu)
            labels = Variable(labels).cuda(args.gpu)
            start_t = time.time() # time.time() returns the current time
            logits = model(images)
            loss = F.cross_entropy(logits, labels)  # Note that the cross_entropy function has already include the softmax function
            print('Val: %s, epo %s/%s, iter %s/%s, %.2f img/sec, loss %.4f, time %s' \
                  % (args.Smodel_name, epo, args.epoch_max, it + 1, len(val_loader), len(names)/(time.time()-start_t), float(loss),
                    datetime.datetime.now().replace(microsecond=0)-start_datetime))
            if accIter['val'] % 1 == 0:
                writer.add_scalar('Validation/loss', loss, accIter['val'])
            view_figure = False  # note that I have not colorized the GT and predictions here
            if accIter['val'] % 100 == 0:
                if view_figure:
                    input_rgb_images = vutils.make_grid(images[:, :3], nrow=8, padding=10)  # can only display 3-channel images, so images[:,:3]
                    writer.add_image('Validation/input_rgb_images', input_rgb_images, accIter['val'])
                    scale = max(1, 255 // args.n_class)  # label (0,1,2..) is invisable, multiply a constant for visualization
                    groundtruth_tensor = labels.unsqueeze(1) * scale  # mini_batch*480*640 -> mini_batch*1*480*640
                    groundtruth_tensor = torch.cat((groundtruth_tensor, groundtruth_tensor, groundtruth_tensor), 1)  # change to 3-channel for visualization
                    groudtruth_images = vutils.make_grid(groundtruth_tensor, nrow=8, padding=10)
                    writer.add_image('Validation/groudtruth_images', groudtruth_images, accIter['val'])
                    predicted_tensor = logits.argmax(1).unsqueeze(1)*scale  # mini_batch*args.n_class*480*640 -> mini_batch*480*640 -> mini_batch*1*480*640
                    predicted_tensor = torch.cat((predicted_tensor, predicted_tensor, predicted_tensor), 1)  # change to 3-channel for visualization, mini_batch*1*480*640
                    predicted_images = vutils.make_grid(predicted_tensor, nrow=8, padding=10)
                    writer.add_image('Validation/predicted_images', predicted_images, accIter['val'])
            accIter['val'] += 1

def testing(epo, Tmodel, Smodel, test_loader):
    Tmodel.eval()
    Smodel.eval()
    conf_total = np.zeros((args.n_class, args.n_class))
    label_list = ["unlabeled", "pothole"]
    testing_results_file = os.path.join(weight_dir, 'testing_results_file.txt')
    with torch.no_grad():
        for it, (images, labels, names) in enumerate(test_loader):
            images = Variable(images).cuda(args.gpu)
            labels = Variable(labels).cuda(args.gpu)
            S_out1,S_out2,S_out3,S_out4,S_out5,Student_logits = Smodel(images)
            label = labels.cpu().numpy().squeeze().flatten()
            prediction = Student_logits.argmax(1).cpu().numpy().squeeze().flatten() # prediction and label are both 1-d array, size: minibatch*640*480
            conf = confusion_matrix(y_true=label, y_pred=prediction, labels=[0,1]) # conf is args.n_class*args.n_class matrix, vertical axis: groundtruth, horizontal axis: prediction
            conf_total += conf
            print('Test: %s, epo %s/%s, iter %s/%s, time %s' % (args.Smodel_name, epo, args.epoch_max, it+1, len(test_loader),
                 datetime.datetime.now().replace(microsecond=0)-start_datetime))
    precision, recall, IoU,F1 = compute_results(conf_total)


    if epo==0:
        with open(testing_results_file, 'w') as f:
            f.write("# %s, initial lr: %s, batch size: %s, date: %s \n" %(args.Smodel_name, args.lr_start, args.batch_size, datetime.date.today()))
            f.write("# epoch: unlabeled, pothole, crack, average(nan_to_num). (Pre %, Acc %, F1 % IoU %)\n")
    with open(testing_results_file, 'a') as f:
        f.write(str(epo)+': ')
        for i in range(len(precision)):
            f.write('%0.4f, %0.4f, %0.4f, %0.4f, ' % (100*precision[i], 100*recall[i], 100*F1[i], 100*IoU[i]))
        f.write('%0.4f, %0.4f, %0.4f, %0.4f\n' % (100*np.mean(np.nan_to_num(precision)), 100*np.mean(np.nan_to_num(recall)), 100*np.mean(np.nan_to_num(F1)), 100*np.mean(np.nan_to_num(IoU))))
    print('saving testing results.')
    with open(testing_results_file, "r") as file:
        writer.add_text('testing_results', file.read().replace('\n', '  \n'), epo)

    ####### test teacher model
    conf_total = np.zeros((args.n_class, args.n_class))
    label_list = ["unlabeled", "pothole"]
    testing_results_file = os.path.join(weight_dir, 'testing_teacher_results_file.txt')
    with torch.no_grad():
        for it, (images, labels, names) in enumerate(test_loader):
            images = Variable(images).cuda(args.gpu)
            labels = Variable(labels).cuda(args.gpu)
            T_out1,T_out2,T_out3,T_out4,T_out5,Teacher_logits = Tmodel(images)
            label = labels.cpu().numpy().squeeze().flatten()
            prediction = Teacher_logits.argmax(1).cpu().numpy().squeeze().flatten() # prediction and label are both 1-d array, size: minibatch*640*480
            conf = confusion_matrix(y_true=label, y_pred=prediction, labels=[0,1]) # conf is args.n_class*args.n_class matrix, vertical axis: groundtruth, horizontal axis: prediction
            conf_total += conf
            print('Test: %s, epo %s/%s, iter %s/%s, time %s' % (args.Tmodel_name, epo, args.epoch_max, it+1, len(test_loader),
                 datetime.datetime.now().replace(microsecond=0)-start_datetime))
    precision, recall, IoU,F1 = compute_results(conf_total)
    writer.add_scalar('Test/average_precision',precision.mean(), epo)
    writer.add_scalar('Test/average_recall', recall.mean(), epo)
    writer.add_scalar('Test/average_IoU', IoU.mean(), epo)
    writer.add_scalar('Test/average_F1', F1.mean(), epo)
    for i in range(len(precision)):
        writer.add_scalar("Test(class)/precision_class_%s" % label_list[i], precision[i], epo)
        writer.add_scalar("Test(class)/recall_class_%s"% label_list[i], recall[i],epo)
        writer.add_scalar('Test(class)/Iou_%s'% label_list[i], IoU[i], epo)
        writer.add_scalar('Test(class)/F1_%s'% label_list[i], F1[i], epo)
    if epo==0:
        with open(testing_results_file, 'w') as f:
            f.write("# %s, initial lr: %s, batch size: %s, date: %s \n" %(args.Tmodel_name, args.lr_start, args.batch_size, datetime.date.today()))
            f.write("# epoch: unlabeled, pothole, crack, average(nan_to_num). (Pre %, Acc %, F1 % IoU %)\n")
    with open(testing_results_file, 'a') as f:
        f.write(str(epo)+': ')
        for i in range(len(precision)):
            f.write('%0.4f, %0.4f, %0.4f, %0.4f, ' % (100*precision[i], 100*recall[i], 100*F1[i], 100*IoU[i]))
        f.write('%0.4f, %0.4f, %0.4f, %0.4f\n' % (100*np.mean(np.nan_to_num(precision)), 100*np.mean(np.nan_to_num(recall)), 100*np.mean(np.nan_to_num(F1)), 100*np.mean(np.nan_to_num(IoU))))
    print('saving testing results.')
    with open(testing_results_file, "r") as file:
        writer.add_text('testing_results', file.read().replace('\n', '  \n'), epo)

if __name__ == '__main__':
   
    #args.gpu = has_empty_GPU(args.gpu,args.need_m,args.sleep)
    torch.cuda.set_device(args.gpu)
    print("\nthe pytorch version:", torch.__version__)
    print("the gpu count:", torch.cuda.device_count())
    print("the current used gpu:", torch.cuda.current_device(), '\n')

    Tmodel = eval(args.Tmodel_name)(n_class=args.n_class)
    Smodel = eval(args.Smodel_name)(n_class=args.n_class)
    
    if args.gpu >= 0: 
        Tmodel.cuda(args.gpu)
        Smodel.cuda(args.gpu)
    optimizer = torch.optim.SGD(Smodel.parameters(), lr=args.lr_start, momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay, last_epoch=-1)

    pretrained_weight = torch.load(args.Teacher_model, map_location = lambda storage, loc: storage.cuda(args.gpu))
    own_state = Tmodel.state_dict()
    for name, param in pretrained_weight.items():
        if name not in own_state:
            continue
        own_state[name].copy_(param)  
    print('done!')

    for name, param in Tmodel.named_parameters():
        param.requires_grad=False


    # preparing folders
    if os.path.exists("./runs"):
        shutil.rmtree("./runs")
    weight_dir = os.path.join("./runs", args.Smodel_name)
    os.makedirs(weight_dir)
    os.chmod(weight_dir, stat.S_IRWXO)  # allow the folder created by docker read, written, and execuated by local machine
 
    writer = SummaryWriter("./runs/tensorboard_log")
    os.chmod("./runs/tensorboard_log", stat.S_IRWXO)  # allow the folder created by docker read, written, and execuated by local machine
    os.chmod("./runs", stat.S_IRWXO) 

    print('training %s on GPU #%d with pytorch' % (args.Smodel_name, args.gpu))
    print('from epoch %d / %s' % (args.epoch_from, args.epoch_max))
    print('weight will be saved in: %s' % weight_dir)

    train_dataset = MF_dataset(data_dir=args.data_dir, split='train', transform=augmentation_methods)
    val_dataset  = MF_dataset(data_dir=args.data_dir, split='validation')
    test_dataset = MF_dataset(data_dir=args.data_dir, split='test')

    train_loader  = DataLoader(
        dataset     = train_dataset,
        batch_size  = args.batch_size,
        shuffle     = True,
        num_workers = args.num_workers,
        pin_memory  = True,
        drop_last   = True
    )
    val_loader  = DataLoader(
        dataset     = val_dataset,
        batch_size  = args.batch_size,
        shuffle     = False,
        num_workers = args.num_workers,
        pin_memory  = True,
        drop_last   = True
    )
    test_loader = DataLoader(
        dataset      = test_dataset,
        batch_size   = args.batch_size,
        shuffle      = False,
        num_workers  = args.num_workers,
        pin_memory   = True,
        drop_last    = True
    )
    start_datetime = datetime.datetime.now().replace(microsecond=0)
    accIter = {'train': 0, 'val': 0}
    for epo in range(args.epoch_from, args.epoch_max):
        print('\ntrain %s, epo #%s begin...' % (args.Smodel_name, epo))
        #scheduler.step() # if using pytorch 0.4.1, please put this statement here 
        train(epo, Tmodel,Smodel, train_loader, optimizer)
        #validation(epo, Smodel, val_loader)

        checkpoint_model_file = os.path.join(weight_dir, str(epo) + '.pth')
        print('saving check point %s: ' % checkpoint_model_file)
        torch.save(Smodel.state_dict(), checkpoint_model_file)

        testing(epo, Tmodel,Smodel, test_loader) # testing is just for your reference, you can comment this line during training
        scheduler.step() # if using pytorch 1.1 or above, please put this statement here
