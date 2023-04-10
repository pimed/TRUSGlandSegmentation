"""
@Author: Sulaiman Vesal
Date: Tuesday, 04, 2020
"""
import os
import argparse
import torch as t
import torch.nn as nn

from datetime import datetime
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

from utils.utils import soft_to_hard_pred
from utils.loss import DiceCoefMultilabelLoss, KnowledgeDistillationLoss
from utils.callbacks import EarlyStoppingCallback, ModelCheckPointCallback
from utils.metric import dice_coef_multilabel
from model.dilated_unet import Segmentation_model
from model.unet_variants import AttU_Net, U_Net, NestedUNet
from dataset.data_generator_npy import *


class Trainer:
    def __init__(self,
                 train_df,
                 test_df,
                 width=160,
                 height=128,  #image size
                 batch_size=64,
                 n_epoch=500,
                 n_classes=2,
                 unet_model=None,
                 unet_model_old=None,
                 device='cuda',
                 trainer_state=None,
                 unet_loss=DiceCoefMultilabelLoss(),
                 unet_lr=0.0001,
                 apply_scheduler=True,  # learning rates
                 gaussian_noise=False,
                 transform=False,
                 r_bright=False,
                 r_gamma=False,
                 n_samples=2000,
                 unet_model_name='unet_model_checkpoint.pt',
                 summary_name='./summary/',
                 channel='channel_first'):

        assert channel == 'channel_first' or channel == 'channel_last', r"channel has to be 'channel_first' or ''channel_last"
        self.train_path, self.test_path = train_df, test_df
        self.WIDTH, self.HEIGHT = width, height
        self.BATCH_SIZE = batch_size
        self.noise = gaussian_noise
        self.epochs = n_epoch
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.unet_model_name = unet_model_name
        self.to_save_entire_model = False # with model structure
        self.file_to_save_summary = summary_name
        self.channel = channel
        self._apply_transform = transform
        self._r_bright = r_bright
        self._r_gamma = r_gamma
        self.unet_model = unet_model
        self.unet_model_old = unet_model_old
        self.unet_loss = unet_loss
        self.unet_lr = unet_lr
        self.apply_scheduler = apply_scheduler
        self.unet_optim  = t.optim.Adam(self.unet_model.parameters(), lr=unet_lr, betas=(0.9, 0.99))
        #self.unet_optim  = t.optim.SGD(self.unet_model.parameters(), lr=0.0001, momentum=0.9, nesterov=True)
        self.device = device

        # Regularization
        # Knowledge distillation loss for feature space
        self.lde = 0.1
        self.lde_flag = self.lde > 0. and self.unet_model_old is not None
        self.lde_loss = nn.MSELoss()
        # Knowledge distillation loss for output space
        self.lkd = 0
        self.lkd_flag = self.lkd > 0. and self.unet_model_old is not None
        self.lkd_loss = KnowledgeDistillationLoss(alpha=0.9)

    def valid_model(self, data_generator, hd=False):
        self.unet_model.eval()
        dice_list = []
        loss_list = []
        reg_loss_list = []
        hd_list = []
        lde = 0.0
        lkd = 0.0
        with t.no_grad():
            for dataset in data_generator:
                # get a bach of validation image and masks
                x_batch, y_batch = dataset
                if self.unet_model_old is not None:
                    with torch.no_grad():
                        # old segmentation model prediction
                        prediction_old, features_old = self.unet_model_old.forward(t.tensor(x_batch).cuda(),
                                                                                   features_out=True)
                # new segmentation model prediction
                prediction, features = self.unet_model.forward(t.tensor(x_batch).cuda(),
                                                               features_out=True)
                # compute the loss
                l = self.unet_loss.forward(predict=prediction,
                                           target=t.tensor(y_batch).cuda(),
                                           numLabels=self.n_classes,
                                           channel='channel_first')

                # Knowledge distillation on features
                if self.lde_flag:
                    lde = self.lde_loss(features, features_old)
                # Knowledge distillation on logits
                if self.lkd_flag:
                    lkd = self.lkd_loss(prediction, prediction_old)
                # Total loss
                loss_tot = l + lde + lkd
                # store the segmentation loss
                loss_list.append(l.item())
                # store the total loss
                reg_loss_list.append(loss_tot.item())
                # convert soft prediction to hard prediction for metric calculation
                y_pred = soft_to_hard_pred(prediction.cpu().detach().numpy(), 1)
                # append the dice multiclass metric for prostate gland
                dice_list.append(dice_coef_multilabel(y_true=y_batch,
                                                      y_pred=y_pred,
                                                      numLabels=self.n_classes,
                                                      channel='channel_first'))

        output = {}
        output["dice"] = np.mean(np.array(dice_list))
        output["loss"] = np.mean(np.array(loss_list))
        output["reg_loss"] = np.mean(np.array(reg_loss_list))
        if hd:
            output["hd"] = np.mean(np.array(hd_list))
        return output

    def get_generators(self, ids_train, ids_valid):

        trainA_generator = DataGenerator(df=ids_train,
                                         x=None,
                                         y=None,
                                         channel="channel_first",
                                         apply_noise=True,
                                         phase="train",
                                         apply_online_aug=True,
                                         batch_size=self.BATCH_SIZE,
                                         n_samples=self.n_samples)

        validA_generator = DataGenerator(df=ids_valid,
                                         x=None,
                                         y=None,
                                         channel="channel_first",
                                         apply_noise=False,
                                         phase="valid",
                                         apply_online_aug=False,
                                         batch_size=self.BATCH_SIZE,
                                         n_samples=-1)
        return iter(trainA_generator), iter(validA_generator)

    def tocude(self):
        self.unet_model.cuda()
        self.unet_loss.cuda()
        if self.unet_model_old is not None:
            self.unet_model_old.cuda()

    def togglephase(self, phase="train"):
        assert phase == "train" or phase == "eval"
        if phase == "train":
            self.unet_model.train()
        else:
            self.unet_model.eval()

    def zerograd(self):
        self.unet_optim.zero_grad()

    def togglegrads(self, model="unet", require_grads=True):
        assert model == "unet"
        if model == "unet":
            for param in self.unet_model.parameters():
                param.requires_grad = require_grads

    def step(self):
        self.unet_optim.step()

    def train_epoch(self, trainA_generator):
        unet_loss = []
        unet_dice = []
        reg_loss = []
        l_reg = torch.tensor(0.)

        self.togglephase(phase="train")
        # train unet
        lde = 0
        lkd = 0
        for dataA in trainA_generator:
            imgA, maskA = dataA
            if self.unet_model_old is not None:
                with torch.no_grad():
                    segmentation_old, features_old = self.unet_model_old.forward(t.tensor(imgA).cuda(),
                                                                                 features_out=True)

            self.zerograd()

            # train the unet model
            self.togglegrads(model="unet",
                             require_grads=True)

            l = t.tensor([0], dtype=t.float32).cuda()
            segmentation, features = self.unet_model.forward(t.tensor(imgA).cuda(),
                                                             features_out=True)
            l_segmentation = self.unet_loss.forward(predict=segmentation,
                                                    target=t.tensor(maskA).cuda(),
                                                    numLabels=self.n_classes,
                                                    channel=self.channel)

            l += l_segmentation
            # distillation on features
            if self.lde_flag:
                lde = self.lde * self.lde_loss(features,
                                               features_old)
            # distillation on logits
            if self.lkd_flag:
                # resize new output to remove new logits and keep only the old ones
                lkd = self.lkd * self.lkd_loss(segmentation,
                                               segmentation_old)
            # total loss including continual learning loss using knowledge distillation
            loss_tot = l+lde+lkd
            # backpropagation the loss
            loss_tot.backward()
            # Update optimizer
            self.step()
            # Log the loss
            reg_loss.append(loss_tot.item())
            unet_loss.append(l_segmentation.item())
            # Convert the model prediction to from soft to hard
            y_pred = soft_to_hard_pred(segmentation.cpu().detach().numpy(), 1)
            # Compute evaluation metric
            unet_dice.append(dice_coef_multilabel(y_true=maskA,
                                                  y_pred=y_pred,
                                                  numLabels=self.n_classes,
                                                  channel=self.channel))
        output = {}
        output["unet_loss"] = np.mean(np.array(unet_loss))
        output["unet_dice"] = np.mean(np.array(unet_dice))
        output["reg_loss"] = np.mean(np.array(reg_loss))
        return output

    def train_model(self, train=True, reg=True, comments=''):

        # create directory for the weights
        root_directory = './weights/' + comments + '/'
        if not os.path.exists(root_directory):
            os.mkdir(root_directory)

        print("Trainining on {} images and validating on {} images...!!".format(len(self.train_path),
                                                                                len(self.test_path)))
        trainA_iterator, validA_iterator = self.get_generators(self.train_path,
                                                               self.test_path)

        # convert models and losses to cuda
        self.tocude()
        if self.apply_scheduler:
            unet_scheduler = ReduceLROnPlateau(optimizer=self.unet_optim,
                                               mode='max',
                                               factor=.1,
                                               patience=15,
                                               verbose=True)

        earlystop = EarlyStoppingCallback(patience=10, mode="max")
        modelcheckpoint_unet = ModelCheckPointCallback(mode="max",
                                                       model_name=root_directory + self.unet_model_name,
                                                       entire_model=self.to_save_entire_model)

        train_loss = []
        train_dice = []
        val_loss = []
        val_dice = []
        val_reg_loss = []
        train_reg_loss= []

        for epoch in range(self.epochs):
            ###################
            # train the model #
            ###################
            if train:
                print("start to train")
                output = self.train_epoch(trainA_iterator)
                train_loss.append(output["unet_loss"])
                train_dice.append(output["unet_dice"])
                train_reg_loss.append(output["reg_loss"])

            ######################
            # validate and test the model #
            ######################
            self.togglephase(phase="eval")

            print("start to valid")
            output = self.valid_model(data_generator=validA_iterator, hd=False)
            val_dice.append(output["dice"])
            val_loss.append(output["loss"])
            val_reg_loss.append(output["reg_loss"])

            # reduceLROnPlateau
            if self.apply_scheduler:
                unet_scheduler.step(metrics=val_dice[-1])

            epoch_len = len(str(self.epochs))

            print_msg_line1 = f'valid_loss: {val_loss[-1]:.5f} '
            print_msg_line2 = f'valid_dice: {val_dice[-1]:.5f} '
            if train:
                print_msg_line1 = f'train_loss: {train_loss[-1]:.5f} ' + print_msg_line1
                print_msg_line2 = f'train_dice: {train_dice[-1]:.5f} ' + print_msg_line2

            if reg:
                print_msg_line1 = print_msg_line1 + f'train_reg_loss: {train_reg_loss[-1]:.5f} ' + \
                                  f'valid_reg_loss: {val_reg_loss[-1]:.5f} '

            print_msg_line1 = f'[{epoch + 1:>{epoch_len}}/{self.epochs:>{epoch_len}}] ' + print_msg_line1
            print_msg_line2 = ' ' * (2 * epoch_len + 4) + print_msg_line2
            print(print_msg_line1)
            print(print_msg_line2)

            # model checkpoint
            monitor_score = val_dice[-1]
            modelcheckpoint_unet.step(monitor=monitor_score, model=self.unet_model, epoch=epoch)

            # early stop
            earlystop.step(val_dice[-1])
            if earlystop.should_stop():
                break
        the_epoch = modelcheckpoint_unet.epoch
        print("Best model on epoch {}: train_dice {}, valid_dice {}".format(the_epoch,
                                                                            train_dice[the_epoch],
                                                                            val_dice[the_epoch]))
        # record train metrics in tensorboard
        writer = SummaryWriter(comment=comments)
        i = 0
        print("write a training summary")
        for t_loss, t_dice, v_loss, v_dice,in zip(
                train_loss, train_dice, val_loss, val_dice):
            writer.add_scalar('Loss/Training', t_loss, i)
            writer.add_scalar('Loss/Validation', v_loss, i)
            i += 1
            if reg:
                i = 1
                for a1, a2 in zip(train_reg_loss, val_reg_loss):
                    writer.add_scalar('t_reg_loss', a1, i)
                    writer.add_scalar('v_reg_loss', a2, i)
                    i += 1
        writer.close()
        print("Finish training")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-base_dir", help = "input image path to nii files", type=str,
                        default="input/images")
    parser.add_argument("--height", help = "input image height", type =int, default=128)
    parser.add_argument("--width", help = "input image width", type=int, default=160)
    parser.add_argument("-unetlr", help="to set the learning rate for the unet", type=float, default=0.0001)
    parser.add_argument("-e", "--epochs", help="the number of epochs to train", type=int, default=50)
    parser.add_argument("-gn", "--gaussianNoise", help="whether to apply gaussian noise", action="store_true",
                        default=True)
    parser.add_argument("--n_samples", help="number of samples to train", type=int, default=-1)
    parser.add_argument("-bs", "--batch_size", help="batch size of training", type=int, default=32)
    parser.add_argument("-nc", "--n_class", help="number of classes to segment", type=int, default=2)
    parser.add_argument("-nf", "--n_filter", help="number of initial filters for DR-UNET", type=int, default=32)
    parser.add_argument("-nb", "--n_block", help="number unet blocks", type=int, default=3)
    parser.add_argument("-cl", "--old_model", help="whether to load previous model for fine-tuning",
                        action="store_true", default=True)
    parser.add_argument("-at", "--attetnion", help="whether to load the model with coordinate attention or not", action="store_true",
                        default=True)
    parser.add_argument("-ow", "--old_weights", help="load the pretrained weights for continual learning", type=str,
                        default="weights/prostateUS.unetcoord_100Per_lesion_LKD_lr_0.0001_32.gaussian_noise/unet_model_checkpoint.pt")
    parser.add_argument("-pt", "--pretrained", help="whether to train from scratch or resume", action="store_true",
                        default=True)
    parser.add_argument("--no_trainCases", help="number of train images to fine-tune", type=int, default=2000)
    parser.add_argument("--no_validCases", help="number of valid images", type=int, default=20000)
    args = parser.parse_args()

    # calculate the comments
    comments = "prostateUS.drunet_UCL_lr_{}_{}".format(args.unetlr, args.n_filter)
    if args.gaussianNoise:
        comments += ".gaussian_noise"
    print(comments)

    # Convert volume nii for US and corresponding segmentation into npy slices on to disk.
    volumes_to_slices(args.base_dir)

    data = []
    for pat in glob.glob(os.path.join(args.base_dir, "npy_train/*.npz")):
        base_name = os.path.split(pat)[-1]
        data.append([pat, base_name])
    train_df = pd.DataFrame(data, columns=['imagePath', 'subject_ID'])
    data = []
    for pat in glob.glob(os.path.join(args.base_dir, "npy_valid/*.npz")):
        base_name = os.path.split(pat)[-1]
        data.append([pat, base_name])
    valid_df = pd.DataFrame(data, columns=['imagePath', 'subject_ID'])

    print("----")
    print("The number of train images:", len(train_df))
    print("The number of valid images:", len(valid_df))

    # Please uncomment this line if you want to train with specific number of train slices
    train_df = train_df.head(args.no_trainCases)
    valid_df = valid_df.head(args.no_validCases)
    print(train_df.head)

    print("The number of train slices:", len(train_df))
    print("The number of valid slices:", len(valid_df))

    if args.old_model:
        print("UNet model trained on UCIL data is loaded...!")
        unet_model_old = Segmentation_model(filters=args.n_filter,
                                            in_channels=3,
                                            n_block=args.n_block,
                                            n_class=args.n_class,
                                            attention = args.attention)
        unet_model_old.load_state_dict(torch.load(args.old_weights))
    else:
        unet_model_old = None

    print("New UNet model is loaded...!")
    unet_model = Segmentation_model(filters=args.n_filter,
                                    in_channels=3,
                                    n_block=args.n_block,
                                    n_class=args.n_class,
                                    attention = args.attention)

    if args.pretrained:
        unet_model.load_state_dict(torch.load(args.old_weights))

    # create trainer class object.
    train_obj = Trainer(train_df,
                        valid_df,
                        width=args.width,
                        height=args.height,
                        batch_size=args.batch_size,
                        unet_model=unet_model,
                        unet_model_old=unet_model_old,
                        unet_loss=DiceCoefMultilabelLoss(),
                        gaussian_noise=args.gaussianNoise,
                        unet_lr=args.unetlr,
                        n_classes=args.n_class,
                        n_epoch=args.epochs,
                        n_samples=args.n_samples)

    # train the models
    print("number of samples {}".format(args.n_samples))
    start = datetime.now()
    t.autograd.set_detect_anomaly(True)
    train_obj.train_model(comments=comments)
    end = datetime.now()
    print("time elapsed for training (hh:mm:ss.ms) {}".format(end - start))
