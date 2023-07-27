import argparse
import os
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
import network
import loss
from torch.utils.data import DataLoader
from data_list import ImageList, ImageList_idx
import random
from randaugment import RandAugmentMC
from torch.nn import DataParallel
from collections import OrderedDict

def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer

def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer

def data_loader(args):
    dsets={}
    dset_loaders={}
    train_bs = args.batch_size
    txt_tar = open(args.t_dset_path).readlines()
    txt_test = open(args.test_dset_path).readlines()
    dsets["target"] = ImageList_idx(txt_tar, transform=TransformFixMatch(mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225)))
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, shuffle=True, num_workers=args.worker,
                                        drop_last=False)
    dsets["test"] = ImageList_idx(txt_test, transform=image_test())
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs * 3, shuffle=False, num_workers=args.worker,
                                      drop_last=False)

    return dset_loaders

def cal_acc_multi(loader, netF_list, netB_list, netC_list, netG_list, args):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            outputs_all = torch.zeros(len(args.src), inputs.shape[0], args.class_num)
            weights_all = torch.ones(inputs.shape[0], len(args.src))
            outputs_all_w = torch.zeros(inputs.shape[0], args.class_num)

            for i in range(len(args.src)):
                features = netB_list[i](netF_list[i](inputs))
                outputs = netC_list[i](features)
                weights = netG_list[i](features)
                outputs_all[i] = outputs
                weights_all[:, i] = weights.squeeze()

            z = torch.sum(weights_all, dim=1)
            z = z + 1e-16

            weights_all = torch.transpose(torch.transpose(weights_all, 0, 1) / z, 0, 1)
            outputs_all = torch.transpose(outputs_all, 0, 1)

            for i in range(inputs.shape[0]):
                outputs_all_w[i] = torch.matmul(torch.transpose(outputs_all[i], 0, 1), weights_all[i])

            if start_test:
                all_output = outputs_all_w.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs_all_w.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(loss.Entropy(nn.Softmax(dim=1)(all_output))).cpu().data.item()
    return accuracy * 100, mean_ent

def msda_regulizer(weak_outputs_all,strong_outputs_all,moment_count):
    weak_outputs_all_ = torch.zeros(len(args.src), weak_outputs_all.shape[1], weak_outputs_all.shape[2])
    strong_outputs_all_ = torch.zeros(len(args.src), strong_outputs_all.shape[1], strong_outputs_all.shape[2])
    outputs_all_ = torch.zeros(len(args.src), weak_outputs_all.shape[1], args.class_num)

    moment1 = torch.tensor(0.0)
    moment2 = torch.tensor(0.0)

    for i in range(len(args.src)):
        strong_outputs_all_[i] =F.log_softmax(strong_outputs_all[i],dim=-1)
        weak_outputs_all_[i]=F.softmax(weak_outputs_all[i],dim=-1)
        outputs_all_mean = weak_outputs_all[i].mean(0)
        outputs_all_[i] = weak_outputs_all[i] - outputs_all_mean

    for i in range(len(args.src)):
        max_probs, targets_u = torch.max(weak_outputs_all_[i], dim=-1)
        mask = max_probs.ge(args.threshold_u).float()
        for j in range(len(args.src)):
            for w in range(len(mask)):
                if mask[w] != 0:
                    moment1 += F.kl_div(strong_outputs_all_[j][w], weak_outputs_all_[i][w].detach(), reduction='sum')

    reg_info = moment1
    return reg_info

def train_target(args):
    dset_loaders = data_loader(args)
    ##set base network
    if args.net[0:3] == 'res':
        netF_list = [network.ResBase(res_name=args.net).cuda() for i in range(len(args.src))]
    elif args.net[0:3] == 'vgg':
        netF_list = [network.VGGBase(vgg_name=args.net).cuda() for i in range(len(args.src))]

    w = 2 * torch.rand((len(args.src),)) - 1

    netB_list = [network.feat_bottleneck(type=args.classifier, feature_dim=netF_list[i].in_features,
                                         bottleneck_dim=args.bottleneck).cuda() for i in range(len(args.src))]
    netC_list = [network.feat_classifier(type=args.layer, class_num=args.class_num, bottleneck_dim=args.bottleneck).cuda() for i
                                        in range(len(args.src))]
    netG_list = [network.scalar(w[i]).cuda() for i in range(len(args.src))]
    
    for i in range(len(args.src)):
        netF_list[i] = DataParallel(netF_list[i], device_ids=[0,1])
        netB_list[i] = DataParallel(netB_list[i], device_ids=[0,1])
        netC_list[i] = DataParallel(netC_list[i], device_ids=[0,1])
        netG_list[i] = DataParallel(netG_list[i], device_ids=[0,1])

    opt_netF=[]
    opt_netB=[]
    opt_netG=[]
    for i in range(len(args.src)):
        modelpath = args.output_dir_src[i] + '/source_F.pt'
        print(modelpath)
        # create new OrderedDict that does not contain module.
        new_state_dict = OrderedDict()
        state_dict = torch.load(modelpath)
        for k, v in state_dict.items():
            name = "module." + k  
            new_state_dict[name] = v  
        
        netF_list[i].load_state_dict(new_state_dict, strict=True)  
        netF_list[i].eval()
        for k, v in netF_list[i].named_parameters():
            opt_netF += [{'params': v, 'lr': args.lr * args.lr_decay1}]

        modelpath = args.output_dir_src[i] + '/source_B.pt'
        print(modelpath)
        new_state_dict = OrderedDict()
        state_dict = torch.load(modelpath)
        for k, v in state_dict.items():
            name = "module." + k  # remove module.
            new_state_dict[name] = v  
        # load params
        netB_list[i].load_state_dict(new_state_dict, strict=True)  
        netB_list[i].eval()
        for k, v in netB_list[i].named_parameters():
            opt_netB += [{'params': v, 'lr': args.lr * args.lr_decay2}]

        modelpath = args.output_dir_src[i] + '/source_C.pt'
        print(modelpath)
        new_state_dict = OrderedDict()
        state_dict = torch.load(modelpath)
        for k, v in state_dict.items():
            name = "module." + k  # remove module.
            new_state_dict[name] = v 
        # load params
        netC_list[i].load_state_dict(new_state_dict, strict=True)  
        netC_list[i].eval()

        for k, v in netC_list[i].named_parameters():
            v.requires_grad = False

        for k, v in netG_list[i].named_parameters():
            opt_netG += [{'params': v, 'lr': args.lr}]

    opt_F=optim.SGD(opt_netF)
    opt_B=optim.SGD(opt_netB)
    opt_G = optim.SGD(opt_netG)
    opt_F=op_copy(opt_F)
    opt_B=op_copy(opt_B)
    opt_G = op_copy(opt_G)

    max_iter = args.max_epoch * len(dset_loaders["target"])
    interval_iter = max_iter // args.interval
    iter_num = 0
    # scalar_num=0
    # writer = SummaryWriter()
    dict_lable = [[] for i in range(len(args.src))]

    #memory bank init
    if args.pl == 'atdoc_na':
        mem_fea = torch.rand(len(args.src),len(dset_loaders["target"].dataset), args.bottleneck).cuda()
        for i in range(len(args.src)):
            mem_fea[i] = mem_fea[i] / torch.norm(mem_fea[i], p=2, dim=1, keepdim=True)
        mem_cls = torch.ones(len(args.src),len(dset_loaders["target"].dataset), args.class_num).cuda() /args.class_num

    while iter_num <max_iter:
        try:
            (weak_img,strong_img),gd_label,tar_idx=iter_test.next()
        except:
            iter_test=iter(dset_loaders["target"])
            (weak_img,strong_img),gd_label,tar_idx=iter_test.next()

        if weak_img.size(0) == 1:
            continue
        weak_img = weak_img.cuda()
        strong_img=strong_img.cuda()

        for i in range(len(args.src)):
            netF_list[i].train()
            netB_list[i].train()

        iter_num+=1
        lr_scheduler(opt_F, iter_num=iter_num, max_iter=max_iter)
        lr_scheduler(opt_B, iter_num=iter_num, max_iter=max_iter)
        lr_scheduler(opt_G, iter_num=iter_num, max_iter=max_iter)

        weak_outputs_all=torch.zeros(len(args.src), weak_img.shape[0], args.class_num)
        strong_outputs_all=torch.zeros(len(args.src), strong_img.shape[0], args.class_num)
        weights_all = torch.ones(weak_img.shape[0], len(args.src))
        outputs_all_w = torch.zeros(weak_img.shape[0], args.class_num)
        weak_features_all = torch.zeros(len(args.src), weak_img.shape[0], args.bottleneck)
        strong_features_all = torch.zeros(len(args.src), strong_img.shape[0], args.bottleneck)

        for i in range(len(args.src)):
            weak_features_test = netB_list[i](netF_list[i](weak_img))
            strong_features_test = netB_list[i](netF_list[i](strong_img))
            weak_features_all[i] = weak_features_test
            strong_features_all[i] = strong_features_test
            weak_outputs_test= netC_list[i](weak_features_test)
            strong_outputs_test=netC_list[i](strong_features_test)
            weights_test = netG_list[i](weak_features_test)
            weak_outputs_all[i]=weak_outputs_test
            strong_outputs_all[i]=strong_outputs_test
            weights_all[:, i] = weights_test.squeeze()

        weak_features_all=weak_features_all.cuda()
        strong_features_all = strong_features_all.cuda()
        weak_outputs_all=weak_outputs_all.cuda()
        strong_outputs_all = strong_outputs_all.cuda()

        classifier_loss = torch.tensor(0.0).cuda()
        eff = iter_num / max_iter
        if args.consis_com > 0:
            loss_msda = args.consis_com * msda_regulizer(weak_outputs_all,strong_outputs_all, len(args.src))/(args.batch_size*args.batch_size)
            classifier_loss += loss_msda

        flag=int(len(args.src)/2)+1
        pre_weight=torch.zeros(len(args.src),weak_img.shape[0])
        pre_all=torch.zeros(len(args.src),weak_img.shape[0])
        pre_lable = torch.zeros(weak_img.shape[0])
        if args.pl == 'atdoc_na':
            for i in range(len(args.src)):
                dis = -torch.mm(weak_features_all[i].detach(), mem_fea[i].t())
                for di in range(dis.size(0)):
                    dis[di, tar_idx[di]] = torch.max(dis)
                _, p1 = torch.sort(dis, dim=1)

                w_na = torch.zeros(weak_features_all[i].size(0), mem_fea[i].size(0)).cuda()
                for wi in range(w_na.size(0)):
                    for wj in range(args.K):
                        w_na[wi][p1[wi, wj]] = 1 / args.K

                weight_, pred = torch.max(w_na.mm(mem_cls[i]), 1)
                pre_weight[i]=weight_
                pre_all[i]=pred

                if iter_num % len(dset_loaders["target"]) == 1:
                    dict_new = dict()
                    for k in range(len(pre_all[i])):
                        if int(pre_all[i][k].item()) in dict_new.keys():
                            dict_new[int(pre_all[i][k].item())]+=1
                        else:
                            dict_new[int(pre_all[i][k].item())] = 1
                    dict_lable[i]=dict_new
                else:
                    for k in range(len(pre_all[i])):
                        if int(pre_all[i][k].item()) in dict_lable[i].keys():
                            dict_lable[i][int(pre_all[i][k].item())]+=1
                        else:
                            dict_lable[i][int(pre_all[i][k].item())] = 1

            pre_all=pre_all.cuda()
            pre_weight=pre_weight.cuda()

            predict_strong=[]
            predict_weak=[]
            true_count = 0
            strong_count=0
            strong_acc=torch.tensor(0.0).cuda()
            use_rate=torch.tensor(0.0).cuda()
            for i in range(weak_img.shape[0]):
                predict_count = {}
                for j in range(len(args.src)):
                    if pre_all[j][i].item() in predict_count.keys():
                        predict_count[pre_all[j][i].item()] += pre_weight[j][i]
                    else:
                        predict_count[pre_all[j][i].item()] = pre_weight[j][i]
                count=0
                max_weight, max_label = max(zip(predict_count.values(),predict_count.keys()))
                if max_weight > args.pl_choice * flag:
                    count=1
                    pre_lable[i]=max_label  
                if count == 1:
                    strong_count=strong_count+1
                    if max_label==gd_label[i]:
                        true_count=true_count+1
                    predict_strong.append(1)
                    predict_weak.append(0)
                else:
                    predict_strong.append(0)
                    predict_weak.append(1)
            predict_strong=torch.tensor(predict_strong).cuda()
            predict_weak=torch.tensor(predict_weak).cuda()
            pre_lable=pre_lable.cuda()

            for i in range(len(args.src)):
                loss_strong = (nn.CrossEntropyLoss(reduction='none')(weak_outputs_all[i], pre_lable.long())) * predict_strong
                cla_loss_str=torch.sum(pre_weight[i] * loss_strong) / (torch.sum(pre_weight[i] * predict_strong).item()+ 1e-16)
                loss_weak = (nn.CrossEntropyLoss(reduction='none')(weak_outputs_all[i], pre_all[i].long())) * predict_weak
                cla_loss_weak = torch.sum(pre_weight[i] * loss_weak) / (torch.sum(pre_weight[i] * predict_weak).item()+ 1e-16)
                cla_loss=cla_loss_str+cla_loss_weak
                classifier_loss += args.cls_par * eff * cla_loss

        z = torch.sum(weights_all, dim=1)
        z = z + 1e-16

        weights_all = torch.transpose(torch.transpose(weights_all, 0, 1) / z, 0, 1).cuda()
        weak_outputs_all = torch.transpose(weak_outputs_all, 0, 1).cuda()

        for i in range(weak_img.shape[0]):
            outputs_all_w[i] = torch.matmul(torch.transpose(weak_outputs_all[i], 0, 1), weights_all[i])

        #IM-loss
        if args.ent:  
            entropy_loss= torch.tensor(0.0)
            softmax_out = nn.Softmax(dim=1)(outputs_all_w)
            entropy_loss = torch.mean(loss.Entropy(softmax_out)) 
            if args.gent:
                msoftmax = softmax_out.mean(dim=0)
                entropy_loss -= torch.sum(-msoftmax * torch.log(msoftmax + 1e-5))
            im_loss = entropy_loss * args.ent_par
            classifier_loss += im_loss

        opt_F.zero_grad()
        opt_B.zero_grad()
        opt_G.zero_grad()
        classifier_loss.backward()
        opt_F.step()
        opt_B.step()
        opt_G.step()

        if args.pl == 'atdoc_na':
            for i in range(len(args.src)):
                netF_list[i].eval()
                netB_list[i].eval()
                with torch.no_grad():
                    features_test = netB_list[i](netF_list[i](weak_img))
                    outputs_test = netC_list[i](features_test)
                    features_test = features_test/torch.norm(features_test, p=2, dim=1, keepdim=True)
                    softmax_out = nn.Softmax(dim=1)(outputs_test)
                    outputs_target = softmax_out ** 2 / ((softmax_out ** 2).sum(dim=0))

                mem_fea[i][tar_idx] =(1.0 - args.momentum) * mem_fea[i][tar_idx] + args.momentum * features_test.clone()
                mem_cls[i][tar_idx] =(1.0 - args.momentum) * mem_cls[i][tar_idx] + args.momentum * outputs_target.clone()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            for i in range(len(args.src)):
                netF_list[i].eval()
                netB_list[i].eval()
            acc, _ = cal_acc_multi(dset_loaders['test'], netF_list, netB_list, netC_list, netG_list, args)
            log_str = 'Iter:{}/{}; Accuracy = {:.2f}%'.format(iter_num, max_iter, acc)
            # print(log_str + '\n')
            for i in range(len(args.src)):
                torch.save(netF_list[i].state_dict(),
                           osp.join(args.output_dir, "target_F_" + str(i) + "_" + args.savename + ".pt"))
                torch.save(netB_list[i].state_dict(),
                           osp.join(args.output_dir, "target_B_" + str(i) + "_" + args.savename + ".pt"))
                torch.save(netC_list[i].state_dict(),
                           osp.join(args.output_dir, "target_C_" + str(i) + "_" + args.savename + ".pt"))
                torch.save(netG_list[i].state_dict(),
                           osp.join(args.output_dir, "target_G_" + str(i) + "_" + args.savename + ".pt"))


class TransformFixMatch(object):
    def __init__(self, mean, std):
        self.weak = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.RandomCrop(size=224),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.125)])
        self.strong = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(size=224),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.125),
            RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)

def image_test(resize_size=256, crop_size=224, alexnet=False):
  if not alexnet:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
  else:
    normalize = transforms.Normalize(meanfile='./ilsvrc_2012_mean.npy')
  return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize
    ])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SHOT')
    # parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--gpu_id', default='0,1,2,3', help="device id to run")
    parser.add_argument('--t', type=int, default=2,
                        help="target")  ## Choose which domain to set as target {0 to len(names)-1}
    parser.add_argument('--max_epoch', type=int, default=60, help="max iterations")
    parser.add_argument('--interval', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=32, help="batch_size")
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    parser.add_argument('--dset', type=str, default='office',choices=['office', 'office-home', 'office-caltech','DomainNet'])
    parser.add_argument('--lr', type=float, default=3 * 1e-4, help="learning rate")
    parser.add_argument('--net', type=str, default='resnet50', help="vgg16, resnet50, res101")
    parser.add_argument('--seed', type=int, default=2021, help="random seed")

    parser.add_argument('--gent', type=bool, default=True)
    parser.add_argument('--ent', type=bool, default=True)
    parser.add_argument('--threshold', type=int, default=0)
    parser.add_argument('--cls_par', type=float, default=1.0)
    parser.add_argument('--ent_par', type=float, default=0.3)
    parser.add_argument('--consis_com', type=float, default=0.5)
    parser.add_argument('--lr_decay1', type=float, default=0.1)
    parser.add_argument('--lr_decay2', type=float, default=1.0)
    parser.add_argument('--K', type=int, default=5)
    parser.add_argument('--T', type=float, default=0.2)
    parser.add_argument('--momentum', type=float, default=0.7)
    parser.add_argument('--pl', type=str, default='atdoc_na', choices=['none', 'atdoc_na', 'atdoc_nc',])
    parser.add_argument('--pl_choice',type=float,default=0.8) 
    parser.add_argument('--threshold_u', default=0.6, type=float,
                        help='consis label threshold_u')

    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--epsilon', type=float, default=1e-5)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--distance', type=str, default='cosine', choices=["euclidean", "cosine"])
    parser.add_argument('--output', type=str, default='ckps/adapt')
    parser.add_argument('--output_src', type=str, default='ckps/source')
    args = parser.parse_args()

    if args.dset == 'office-home':
        names = ['Art', 'Clipart', 'Product', 'Real_World']
        args.class_num = 65
    if args.dset == 'office':
        names = ['amazon', 'dslr', 'webcam']
        args.class_num = 31
    if args.dset == 'office-caltech':
        names = ['amazon', 'caltech', 'dslr', 'webcam']
        args.class_num = 10
    if args.dset == 'DomainNet':
        names = ['clipart','infograph','painting','quickdraw','real','sketch']
        args.class_num = 346

    args.src = []
    for i in range(len(names)):
        if i == args.t:
            continue
        else:
            args.src.append(names[i])

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    # torch.cuda.set_device(1)

    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    for i in range(len(names)):
        if i != args.t:
            continue
        folder = './data/'
        args.t_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'
        args.test_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'
        print(args.t_dset_path)

    args.output_dir_src = []
    for i in range(len(args.src)):
        args.output_dir_src.append(osp.join(args.output_src, args.dset, args.src[i][0].upper()))
    print(args.output_dir_src)
    args.output_dir = osp.join(args.output, args.dset, names[args.t][0].upper())

    if not osp.exists(args.output_dir):
        os.system('mkdir -p ' + args.output_dir)
    if not osp.exists(args.output_dir):
        os.mkdir(args.output_dir)

    args.savename = 'par_' + str(args.cls_par)

    train_target(args)
