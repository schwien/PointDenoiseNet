from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Variable
import utils


class STN(nn.Module):
    def __init__(self, num_scales=1, num_points=500, dim=3, sym_op='max', quaternion =False):
        super(STN, self).__init__()
        self.quaternion = quaternion
        self.dim = dim
        self.sym_op = sym_op
        self.num_scales = num_scales
        self.num_points = num_points

        self.conv1 = torch.nn.Conv1d(self.dim, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.mp1 = torch.nn.MaxPool1d(num_points)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        if not quaternion:
            self.fc3 = nn.Linear(256, self.dim*self.dim)
        else:
            self.fc3 = nn.Linear(256, 4)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        if self.num_scales > 1:
            self.fc0 = nn.Linear(1024*self.num_scales, 1024)
            self.bn0 = nn.BatchNorm1d(1024)


    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # symmetric operation over all points
        if self.num_scales == 1:
            x = self.mp1(x)
        else:
            if x.is_cuda:
                x_scales = Variable(torch.cuda.FloatTensor(x.size(0), 1024*self.num_scales, 1))
            else:
                x_scales = Variable(torch.FloatTensor(x.size(0), 1024*self.num_scales, 1))
            for s in range(self.num_scales):
                x_scales[:, s*1024:(s+1)*1024, :] = self.mp1(x[:, :, s*self.num_points:(s+1)*self.num_points])
            x = x_scales

        x = x.view(-1, 1024*self.num_scales)

        if self.num_scales > 1:
            x = F.relu(self.bn0(self.fc0(x)))

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        if not self.quaternion:
            iden = Variable(torch.from_numpy(np.identity(self.dim, 'float32')).clone()).view(1, self.dim*self.dim).repeat(batchsize, 1)

            if x.is_cuda:
                iden = iden.cuda()
            x = x + iden
            x = x.view(-1, self.dim, self.dim)
        else:
            # add identity quaternion (so the network can output 0 to leave the point cloud identical)
            iden = Variable(torch.FloatTensor([1, 0, 0, 0]))
            if x.is_cuda:
                iden = iden.cuda()
            x = x + iden

            # convert quaternion to rotation matrix
            if x.is_cuda:
                trans = Variable(torch.cuda.FloatTensor(batchsize, 3, 3))
            else:
                trans = Variable(torch.FloatTensor(batchsize, 3, 3))
            x = utils.batch_quat_to_rotmat(x, trans)
        return x


class PointNetfeat(nn.Module):
    def __init__(self, num_scales=1, num_points=500, use_point_stn=True, use_feat_stn=True, sym_op='max', get_pointfvals=False, point_tuple=1):
        super(PointNetfeat, self).__init__()
        self.num_points = num_points
        self.num_scales = num_scales
        self.use_point_stn = use_point_stn
        self.use_feat_stn = use_feat_stn
        self.sym_op = sym_op
        self.get_pointfvals = get_pointfvals
        self.point_tuple = point_tuple

        if self.use_point_stn:
            # self.stn1 = STN(num_scales=self.num_scales, num_points=num_points, dim=3, sym_op=self.sym_op)
            self.stn1 = STN(num_scales=self.num_scales, num_points=num_points*self.point_tuple, dim=3, sym_op=self.sym_op, quaternion = True)

        if self.use_feat_stn:
            self.stn2 = STN(num_scales=self.num_scales, num_points=num_points, dim=64, sym_op=self.sym_op)

        self.conv0a = torch.nn.Conv1d(3*self.point_tuple, 64, 1)
        self.conv0b = torch.nn.Conv1d(64, 64, 1)

        # TODO remove
        # self.conv0c = torch.nn.Conv1d(64, 64, 1)
        # self.bn0c = nn.BatchNorm1d(64)
        # self.conv1b = torch.nn.Conv1d(64, 64, 1)
        # self.bn1b = nn.BatchNorm1d(64)


        self.bn0a = nn.BatchNorm1d(64)
        self.bn0b = nn.BatchNorm1d(64)
        self.conv1 = torch.nn.Conv1d(64, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        if self.num_scales > 1:
            self.conv4 = torch.nn.Conv1d(1024, 1024*self.num_scales, 1)
            self.bn4 = nn.BatchNorm1d(1024*self.num_scales)

        if self.sym_op == 'max':
            self.mp1 = torch.nn.MaxPool1d(num_points)
        elif self.sym_op == 'sum':
            self.mp1 = None
        else:
            raise ValueError('Unsupported symmetric operation: %s' % (self.sym_op))

    def forward(self, x):

        # input transform
        if self.use_point_stn:
            # from tuples to list of single points
            x = x.view(x.size(0), 3, -1)
            trans = self.stn1(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans)
            x = x.transpose(2, 1)
            x = x.contiguous().view(x.size(0), 3*self.point_tuple, -1)
        else:
            trans = None

        # mlp (64,64)
        x = F.relu(self.bn0a(self.conv0a(x)))
        x = F.relu(self.bn0b(self.conv0b(x)))
        # TODO remove
        #x = F.relu(self.bn0c(self.conv0c(x)))

        # feature transform
        if self.use_feat_stn:
            trans2 = self.stn2(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans2)
            x = x.transpose(2, 1)
        else:
            trans2 = None

        # mlp (64,128,1024)
        x = F.relu(self.bn1(self.conv1(x)))
        # TODO remove
        #x = F.relu(self.bn1b(self.conv1b(x)))

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))

        # mlp (1024,1024*num_scales)
        if self.num_scales > 1:
            x = self.bn4(self.conv4(F.relu(x)))

        if self.get_pointfvals:
            pointfvals = x
        else:
            pointfvals = None # so the intermediate result can be forgotten if it is not needed

        # symmetric max operation over all points
        if self.num_scales == 1:
            if self.sym_op == 'max':
                x = self.mp1(x)
            elif self.sym_op == 'sum':
                x = torch.sum(x, 2, keepdim=True)
            else:
                raise ValueError('Unsupported symmetric operation: %s' % (self.sym_op))

        else:
            if x.is_cuda:
                x_scales = Variable(torch.cuda.FloatTensor(x.size(0), 1024*self.num_scales**2, 1))
            else:
                x_scales = Variable(torch.FloatTensor(x.size(0), 1024*self.num_scales**2, 1))
            if self.sym_op == 'max':
                for s in range(self.num_scales):
                    x_scales[:, s*self.num_scales*1024:(s+1)*self.num_scales*1024, :] = self.mp1(x[:, :, s*self.num_points:(s+1)*self.num_points])
            elif self.sym_op == 'sum':
                for s in range(self.num_scales):
                    x_scales[:, s*self.num_scales*1024:(s+1)*self.num_scales*1024, :] = torch.sum(x[:, :, s*self.num_points:(s+1)*self.num_points], 2, keepdim=True)
            else:
                raise ValueError('Unsupported symmetric operation: %s' % (self.sym_op))
            x = x_scales

        x = x.view(-1, 1024*self.num_scales**2)

        return x, trans, trans2, pointfvals


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, conv = False):
        super(BasicBlock, self).__init__()
        if conv:
            self.l1 = torch.nn.Conv1d(in_planes, planes, 1)
            self.l2 = torch.nn.Conv1d(planes, planes, 1)
        else:
            self.l1 = nn.Linear(in_planes,planes)
            self.l2 = nn.Linear(planes, planes)

        stdv = 0.001 # for working small initialisation
        # self.l1.weight.data.uniform_(-stdv, stdv)

        self.l1.weight.data.uniform_(-stdv, stdv)
        self.l2.weight.data.uniform_(-stdv, stdv)
        self.l1.bias.data.uniform_(-stdv, stdv)
        self.l2.bias.data.uniform_(-stdv, stdv)

        self.bn1 = nn.BatchNorm1d(planes, momentum = 0.01)
        self.shortcut = nn.Sequential()
        if in_planes != planes:
            if conv:
                self.l0 = nn.Conv1d(in_planes, planes, 1)
            else:
                self.l0 = nn.Linear(in_planes, planes)

            self.l0.weight.data.uniform_(-stdv, stdv)
            self.l0.bias.data.uniform_(-stdv, stdv)

            self.shortcut = nn.Sequential(self.l0,nn.BatchNorm1d(planes))
        self.bn2 = nn.BatchNorm1d(planes, momentum = 0.01)

    def forward(self, x):
            out = F.relu(self.bn1(self.l1(x)))
            out = self.bn2(self.l2(out))
            out += self.shortcut(x)  #实现了跳跃连接
            out = F.relu(out)
            return out


class ResSTN(nn.Module):
    def __init__(self, num_scales=1, num_points=500, dim=3, sym_op='max', quaternion =False):
        super(ResSTN, self).__init__()
        self.quaternion = quaternion
        self.dim = dim
        self.sym_op = sym_op
        self.num_scales = num_scales
        self.num_points = num_points

        self.b1 = BasicBlock(self.dim, 64, conv = True)
        self.b2 = BasicBlock(64, 128, conv = True)
        self.b3 = BasicBlock(128, 1024, conv = True)
        self.mp1 = torch.nn.MaxPool1d(num_points)

        self.bfc1 = BasicBlock(1024, 512)
        self.bfc2 = BasicBlock(512, 256)
        if not quaternion:
            self.bfc3 = BasicBlock(256, self.dim*self.dim)
        else:
            self.bfc3 = BasicBlock(256, 4)

        if self.num_scales > 1:
            self.bfc0 = BasicBlock(1024*self.num_scales, 1024)


    def forward(self, x):
        batchsize = x.size()[0]
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        # symmetric operation over all points
        if self.num_scales == 1:
            x = self.mp1(x)
        else:
            if x.is_cuda:
                x_scales = Variable(torch.cuda.FloatTensor(x.size(0), 1024*self.num_scales, 1))
            else:
                x_scales = Variable(torch.FloatTensor(x.size(0), 1024*self.num_scales, 1))
            for s in range(self.num_scales):
                x_scales[:, s*1024:(s+1)*1024, :] = self.mp1(x[:, :, s*self.num_points:(s+1)*self.num_points])
            x = x_scales

        x = x.view(-1, 1024*self.num_scales)

        if self.num_scales > 1:
            x = self.bfc0(x)

        x =self.bfc1(x)
        x = self.bfc2(x)
        x = self.bfc3(x)

        if not self.quaternion:
            iden = Variable(torch.from_numpy(np.identity(self.dim, 'float32')).clone()).view(1, self.dim*self.dim).repeat(batchsize, 1)

            if x.is_cuda:
                iden = iden.cuda()
            x = x + iden
            x = x.view(-1, self.dim, self.dim)
        else:
            # add identity quaternion (so the network can output 0 to leave the point cloud identical)
            iden = Variable(torch.FloatTensor([1, 0, 0, 0]))
            if x.is_cuda:
                iden = iden.cuda()
            x = x + iden

            # convert quaternion to rotation matrix
            if x.is_cuda:
                trans = Variable(torch.cuda.FloatTensor(batchsize, 3, 3))
            else:
                trans = Variable(torch.FloatTensor(batchsize, 3, 3))
            x = utils.batch_quat_to_rotmat(x, trans)
        return x

class ResPointNetfeat(nn.Module):
    def __init__(self, num_scales=1, num_points=500, use_point_stn=True, use_feat_stn=True, sym_op='max', get_pointfvals=False, point_tuple=1):
        super(ResPointNetfeat, self).__init__()
        self.num_points = num_points
        self.num_scales = num_scales
        self.use_point_stn = use_point_stn
        self.use_feat_stn = use_feat_stn
        self.sym_op = sym_op
        self.get_pointfvals = get_pointfvals
        self.point_tuple = point_tuple

        if self.use_point_stn:
            # self.stn1 = STN(num_scales=self.num_scales, num_points=num_points, dim=3, sym_op=self.sym_op)
            self.stn1 = ResSTN(num_scales=self.num_scales, num_points=num_points*self.point_tuple, dim=3, sym_op=self.sym_op, quaternion=True)

        if self.use_feat_stn:
            self.stn2 = ResSTN(num_scales=self.num_scales, num_points=num_points, dim=64, sym_op=self.sym_op)

        self.b0a = BasicBlock(3*self.point_tuple, 64, conv = True)
        self.b0b = BasicBlock(64, 64, conv=True)

        self.b1 = BasicBlock(64, 64, conv = True)
        self.b2 = BasicBlock(64, 128, conv = True)
        self.b3 = BasicBlock(128, 1024, conv = True)

        if self.num_scales > 1:
            self.b4 = BasicBlock(1024, 1024*self.num_scales, conv = True)

        if self.sym_op == 'max':
            self.mp1 = torch.nn.MaxPool1d(num_points)
        elif self.sym_op == 'sum':
            self.mp1 = None
        else:
            raise ValueError('Unsupported symmetric operation: %s' % (self.sym_op))

    def forward(self, x):

        # input transform
        if self.use_point_stn:
            # from tuples to list of single points
            x = x.view(x.size(0), 3, -1)
            trans = self.stn1(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans)
            x = x.transpose(2, 1)
            x = x.contiguous().view(x.size(0), 3*self.point_tuple, -1)
        else:
            trans = None

        # mlp (64,64)
        x = self.b0a(x)
        x = self.b0b(x)

        # feature transform
        if self.use_feat_stn:
            trans2 = self.stn2(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans2)
            x = x.transpose(2, 1)
        else:
            trans2 = None

        # mlp (64,128,1024)
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)

        # mlp (1024,1024*num_scales)
        if self.num_scales > 1:
            x = self.b4(x)

        if self.get_pointfvals:
            pointfvals = x
        else:
            pointfvals = None # so the intermediate result can be forgotten if it is not needed

        # symmetric max operation over all points
        if self.num_scales == 1:
            if self.sym_op == 'max':
                x = self.mp1(x)
            elif self.sym_op == 'sum':
                x = torch.sum(x, 2, keepdim=True)
            else:
                raise ValueError('Unsupported symmetric operation: %s' % (self.sym_op))

        else:
            if x.is_cuda:
                x_scales = Variable(torch.cuda.FloatTensor(x.size(0), 1024*self.num_scales**2, 1))
            else:
                x_scales = Variable(torch.FloatTensor(x.size(0), 1024*self.num_scales**2, 1))
            if self.sym_op == 'max':
                for s in range(self.num_scales):
                    x_scales[:, s*self.num_scales*1024:(s+1)*self.num_scales*1024, :] = self.mp1(x[:, :, s*self.num_points:(s+1)*self.num_points])
            elif self.sym_op == 'sum':
                for s in range(self.num_scales):
                    x_scales[:, s*self.num_scales*1024:(s+1)*self.num_scales*1024, :] = torch.sum(x[:, :, s*self.num_points:(s+1)*self.num_points], 2, keepdim=True)
            else:
                raise ValueError('Unsupported symmetric operation: %s' % (self.sym_op))
            x = x_scales

        x = x.view(-1, 1024*self.num_scales**2)

        return x, trans, trans2, pointfvals


class ResPCPNet(nn.Module):
    def __init__(self, num_points=500, output_dim=3, use_point_stn=True, use_feat_stn=True, sym_op='max', get_pointfvals=False, point_tuple=1):
        super(ResPCPNet, self).__init__()
        self.num_points = num_points

        self.feat = ResPointNetfeat(
            num_points=num_points,
            num_scales=1,
            use_point_stn=use_point_stn,
            use_feat_stn=use_feat_stn,
            sym_op=sym_op,
            get_pointfvals=get_pointfvals,
            point_tuple=point_tuple)

        self.b1 = BasicBlock(1024, 512)

        self.b2 = BasicBlock(512, 256)
        #self.b3 = BasicBlock(256, output_dim)
		
        self.fc1_1 = nn.Linear(1024,128*512)
        self.fc2_1 = nn.Linear(512,64*128)#nn.Linear(512,64*256) !
        self.fc3_1 = nn.Linear(256,64*3)
        
#        self.bn1 = nn.BatchNorm1d(1024)
#        self.bn2 = nn.BatchNorm1d(512)
#        self.bn3 = nn.BatchNorm1d(256)#nn.BatchNorm1d(64*256) !
#        self.bn4 = nn.BatchNorm1d(128*512)#nn.BatchNorm1d(256)
#        self.bn5 = nn.BatchNorm1d(64*128)
#        
        self.conv1_1 = torch.nn.Conv1d(512,512,1)#torch.nn.Conv1d(256,256,1) !
        self.conv1_2 = torch.nn.Conv1d(512,256,1)
        self.conv1_3 = torch.nn.Conv1d(256,int((512*3)/128),1)
        self.conv2_1 = torch.nn.Conv1d(128,6,1)#torch.nn.Conv1d(256,12,1) !
        self.maxpool=torch.nn.MaxPool1d(512)


    def forward(self, x):
        x_1, trans, trans2, pointfvals = self.feat(x)#1024
        x_2 = self.b1(x_1)#512
        x_3 = self.b2(x_2)#256
		
        pc1_feat = self.fc3_1(x_3)
        pc1_xyz = pc1_feat.reshape(-1,64,3) #64x3 center1
        
        pc2_feat = F.relu(self.fc2_1(x_2))
        pc2_feat = pc2_feat.reshape(-1,128,64)
        pc2_xyz =self.conv2_1(pc2_feat) #6x64 center2
        
        pc3_feat = F.relu(self.fc1_1(x_1))
        pc3_feat = pc3_feat.reshape(-1,512,128)
        pc3_feat = F.relu(self.conv1_1(pc3_feat))
        pc3_feat = F.relu(self.conv1_2(pc3_feat))
        pc3_xyz = self.conv1_3(pc3_feat) #12x128 fine
        
        pc1_xyz_expand = torch.unsqueeze(pc1_xyz,2)
        pc2_xyz = pc2_xyz.transpose(1,2)
        pc2_xyz = pc2_xyz.reshape(-1,64,2,3)
        pc2_xyz = pc1_xyz_expand+pc2_xyz
        pc2_xyz = pc2_xyz.reshape(-1,128,3) 
        
        pc2_xyz_expand = torch.unsqueeze(pc2_xyz,2)
        pc3_xyz = pc3_xyz.transpose(1,2)
        pc3_xyz = pc3_xyz.reshape(-1,128,int(512/128),3)
        pc3_xyz = pc2_xyz_expand+pc3_xyz
        pc3_xyz = pc3_xyz.reshape(-1,512,3) 
        #x = self.b3(x)
        x = pc3_xyz.transpose(1,2)
        x = self.maxpool(x)
        x = x.view(-1,3)
        return x, trans, trans2, pointfvals

class ResMSPCPNet(nn.Module):
    def __init__(self, num_scales=2, num_points=500, output_dim=3, use_point_stn=True, use_feat_stn=True, sym_op='max', get_pointfvals=False, point_tuple=1):
        super(ResMSPCPNet, self).__init__()
        self.num_points = num_points

        self.feat = ResPointNetfeat(
            num_points=num_points,
            num_scales=num_scales,
            use_point_stn=use_point_stn,
            use_feat_stn=use_feat_stn,
            sym_op=sym_op,
            get_pointfvals=get_pointfvals,
            point_tuple=point_tuple)
        self.b0 = BasicBlock(1024*num_scales**2, 1024)
        self.b1 = BasicBlock(1024, 512)
        self.b2 = BasicBlock(512, 256)
        self.b3 = BasicBlock(256, output_dim)

    def forward(self, x):
        x, trans, trans2, pointfvals = self.feat(x)
        x = self.b0(x)
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        return x, trans, trans2, pointfvals

class PCPNet(nn.Module):
    def __init__(self, num_points=500, output_dim=3, use_point_stn=True, use_feat_stn=True, sym_op='max', get_pointfvals=False, point_tuple=1):
        super(PCPNet, self).__init__()
        self.num_points = num_points

        self.feat = PointNetfeat(
            num_points=num_points,
            num_scales=1,
            use_point_stn=use_point_stn,
            use_feat_stn=use_feat_stn,
            sym_op=sym_op,
            get_pointfvals=get_pointfvals,
            point_tuple=point_tuple)
        self.fc1 = nn.Linear(1024, 512)
        #self.fc_new = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, output_dim)
        self.bn1 = nn.BatchNorm1d(512)
        #self.bn_new = nn.BatchNorm1d(512)

        self.bn2 = nn.BatchNorm1d(256)
        self.do1 = nn.Dropout(p=0.3)
        #self.do_new = nn.Dropout(p=0.3)

        self.do2 = nn.Dropout(p=0.3)
    def forward(self, x):
        x, trans, trans2, pointfvals = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.do1(x)

        # x = F.relu(self.bn_new(self.fc_new(x)))
        #x = self.do_new(x)

        x = F.relu(self.bn2(self.fc2(x)))
        x = self.do2(x)
        x = self.fc3(x)

        return x, trans, trans2, pointfvals

class MSPCPNet(nn.Module):
    def __init__(self, num_scales=2, num_points=500, output_dim=3, use_point_stn=True, use_feat_stn=True, sym_op='max', get_pointfvals=False, point_tuple=1):
        super(MSPCPNet, self).__init__()
        self.num_points = num_points

        self.feat = PointNetfeat(
            num_points=num_points,
            num_scales=num_scales,
            use_point_stn=use_point_stn,
            use_feat_stn=use_feat_stn,
            sym_op=sym_op,
            get_pointfvals=get_pointfvals,
            point_tuple=point_tuple)
        self.fc0 = nn.Linear(1024*num_scales**2, 1024)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, output_dim)
        self.bn0 = nn.BatchNorm1d(1024)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.do1 = nn.Dropout(p=0.3)
        self.do2 = nn.Dropout(p=0.3)
    def forward(self, x):
        x, trans, trans2, pointfvals = self.feat(x)
        x = F.relu(self.bn0(self.fc0(x)))
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.do1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.do2(x)
        x = self.fc3(x)

        return x, trans, trans2, pointfvals
