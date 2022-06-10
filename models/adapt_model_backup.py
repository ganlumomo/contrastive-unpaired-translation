import numpy as np
import torch
from .base_model import BaseModel
from . import networks
from .patchnce import PatchNCELoss
import util.util as util
import torch.nn as nn
import torch.nn.functional as F

class AdaptModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """  Configures options specific for CUT model
        """
        parser.add_argument('--CUT_mode', type=str, default="CUT", choices='(CUT, cut, FastCUT, fastcut, ADAPT, adapt)')
        parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight for GAN loss：GAN(G(X))')
        parser.add_argument('--lambda_NCE', type=float, default=1.0, help='weight for NCE loss: NCE(G(X), X)')
        parser.add_argument('--nce_idt', type=util.str2bool, nargs='?', const=True, default=False, help='use NCE loss for identity mapping: NCE(G(Y), Y))')
        parser.add_argument('--nce_layers', type=str, default='0,4,8,12,16', help='compute NCE loss on which layers')
        parser.add_argument('--nce_includes_all_negatives_from_minibatch',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help='(used for single image translation) If True, include the negatives from the other samples of the minibatch when computing the contrastive loss. Please see models/patchnce.py for more details.')
        parser.add_argument('--netF', type=str, default='mlp_sample', choices=['sample', 'reshape', 'mlp_sample'], help='how to downsample the feature map')
        parser.add_argument('--netF_nc', type=int, default=256)
        parser.add_argument('--nce_T', type=float, default=0.07, help='temperature for NCE loss')
        parser.add_argument('--num_patches', type=int, default=256, help='number of patches per layer')
        parser.add_argument('--flip_equivariance',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help="Enforce flip-equivariance as additional regularization. It's used by FastCUT, but not CUT")

        parser.set_defaults(pool_size=0)  # no image pooling

        opt, _ = parser.parse_known_args()
        
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        # specify the training losses you want to print out.
        # The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_SEG', 'G_GAN', 'D_source', 'D_target', 'G']
        self.visual_names = ['source', 'source_pred', 'source_label', 'target', 'target_pred', 'target_label']

        if self.isTrain:
            self.model_names = ['G', 'F', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']

        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, self.gpu_ids, opt)
        self.netF = networks.define_F(opt.input_nc, opt.netF, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)

        if self.isTrain:
            self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)

            # define loss functions
            self.criterionSEG = networks.CrossEntropy2d().to(self.device)
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            
            # define optimizers
            self.optimizer_G = torch.optim.SGD(self.optim_parameters(), lr=opt.lr, momentum=0.9, weight_decay=0.0001)
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=0.0002, betas=(opt.beta1, opt.beta2))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def optim_parameters(self):
        return [{'params': self.netG.backbone.parameters(), 'lr': self.opt.lr},
                {'params': self.netG.classifier.parameters(), 'lr': 10 * self.opt.lr}]

    def lr_poly(self, base_lr, iters, n_iters, power):
        return base_lr * ((1-float(iters) / n_iters) ** (power))

    def adjust_learning_rate(self, iters):
        lr = self.lr_poly(self.opt.lr, iters, self.opt.n_iters, power=0.9)
        self.optimizer_G.param_groups[0]['lr'] = lr
        if len(self.optimizer_G.param_groups) > 1:
            self.optimizer_G.param_groups[1]['lr'] = lr * 10

    def data_dependent_initialize(self, data):
        """
        The feature network netF is defined in terms of the shape of the intermediate, extracted
        features of the encoder portion of netG. Because of this, the weights of netF are
        initialized at the first feedforward pass with some input images.
        Please also see PatchSampleF.create_mlp(), which is called at the first forward() call.
        """
        bs_per_gpu = data["A"].size(0) // max(len(self.opt.gpu_ids), 1)
        self.set_input(data)
        self.source = self.source[:bs_per_gpu]
        self.source_label = self.source_label[:bs_per_gpu]
        self.target = self.target[:bs_per_gpu]
        self.target_label = self.target_label[:bs_per_gpu]
        self.forward()                     # compute fake images: G(A)
        if self.opt.isTrain:
            self.compute_D_loss().backward()                  # calculate gradients for D
            self.compute_G_loss().backward()                   # calculate graidents for G
    
    def optimize_parameters(self):
        # forward
        self.forward()

        # update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.loss_D = self.compute_D_loss()
        self.loss_D.backward()
        self.optimizer_D.step()

        # update G
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.loss_G = self.compute_G_loss()
        self.loss_G.backward()
        self.optimizer_G.step()

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
        self.source = input['A'].to(self.device)
        self.source_label = input['A_label'].to(self.device)
        self.target = input['B'].to(self.device)
        self.target_label = input['B_label'].to(self.device)
        self.image_paths = input['B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        if self.opt.isTrain:
            self.image = torch.cat((self.source, self.target), dim=0)
            self.pred = self.netG(self.image)
            self.source_pred = self.pred[:self.source.size(0)]
            self.target_pred = self.pred[self.source.size(0):]
        else:
            self.target_pred = self.netG(self.target)

    def compute_D_loss(self):
        """Calculate GAN loss for the discriminator"""
        source_pred = self.source_pred.detach()
        target_pred = self.target_pred.detach()
        # Source; stop backprop to the generator by detaching
        pred_source = self.netD(F.softmax(source_pred))
        self.loss_D_source = self.criterionGAN(pred_source, False).mean()
        # Target
        pred_target = self.netD(F.softmax(target_pred))
        self.loss_D_target = self.criterionGAN(pred_target, True).mean()

        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_source + self.loss_D_target) * 0.5
        return self.loss_D

    def compute_G_loss(self):
        """Calculate GAN and NCE loss for the generator"""
        target_pred = self.target_pred
        source_pred = self.source_pred
        self.loss_G_SEG = self.criterionSEG(source_pred, self.source_label)
                #+ self.criterionSEG(target_pred, self.target_label) * 2
        
        # First, G(A) should fake the discriminator
        if self.opt.lambda_GAN > 0.0:
            pred_target = self.netD(F.softmax(target_pred))
            loss_G_target = self.criterionGAN(pred_target, False).mean()
            #pred_source = self.netD(F.softmax(source_pred))
            #loss_G_source = self.criterionGAN(pred_source, True).mean()
            self.loss_G_GAN = loss_G_target
        else:
            self.loss_G_GAN = 0.0

        self.loss_G = self.loss_G_SEG + self.loss_G_GAN * self.opt.lambda_GAN
        return self.loss_G
