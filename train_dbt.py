# From "Learning Deep Bilinear Transformation for Fine-grained Image Represetation"
# https://dl.acm.org/doi/pdf/10.5555/3454287.3454672
#
# @incollection{NIPS2019_8680,
# title = {Learning Deep Bilinear Transformation for Fine-grained Image Representation},
# author = {Zheng, Heliang and Fu, Jianlong and Zha, Zheng-Jun and Luo, Jiebo},
# booktitle = {Advances in Neural Information Processing Systems 32},
# pages = {4279--4288},
# year = {2019}
#
# Original MXNet version https://github.com/researchmm/DBTNet
# Modifications of Pytorch implementation by https://github.com/wuwusky/DBT_Net
#

import argparse
import collections
import math
import os
import torch
import torch.nn.parallel
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim

import torch.utils.data
from torch.utils.data import default_collate
import torch.utils.data.distributed
from torchvision import transforms, datasets, models
from torchvision.transforms import v2

import torch.nn as nn
from PIL import Image

# from softtriple import loss
# from softtriple.evaluation import evaluation
# from softtriple import net

import dbt_pytorch as dbt
from timm.data.auto_augment import rand_augment_transform
from timm.data.transforms import RandomResizedCropAndInterpolation


def train(train_loader, model, loss_fcn, g_lam, optimizer, args, num_avg_iter=500):
    # switch to train mode

    model.train()
    if args.freeze_BN:
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    run_loss = 0
    for i, (input, target) in enumerate(train_loader):

        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output, loss_sg = model(input)
        # loss_fcn.loss_sg = loss_sg
        loss = loss_fcn(output, target) + torch.sum(loss_sg) * g_lam
        run_loss += loss.item()

        # print("i {} t loss {}".format(i, run_loss))
        if (i % num_avg_iter == 0) and i > 0:
            print(
                "Training loss batch {} running avg {} cur {} group {}".format(
                    i, float(run_loss) / num_avg_iter, loss.item(), loss_sg.item()
                )
            )
            run_loss = 0

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def validate(val_loader, model, loss_fcn, g_lam, epoch, args):
    # Validation phase
    model.eval()
    val_loss = 0
    correct = 0
    n_val = 0
    with torch.no_grad():
        for data in val_loader:
            inputs, labels = data
            if args.gpu is not None:
                inputs = inputs.cuda(args.gpu, non_blocking=True)
            labels = labels.cuda(args.gpu, non_blocking=True)

            outputs, loss_sg = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            predicted = torch.argmax(probs, dim=1)
            # print(probs)
            # _, predicted = torch.max(outputs.data, 1)
            # print("predicted {} labels {}".format(predicted, labels))
            A = predicted == labels
            # print('A {} l i {}'.format(A, len(inputs)))
            correct += A.sum().item()

            # just for statistics
            # loss_fcn.loss_sg = loss_sg
            tloss = loss_fcn(outputs, labels) + torch.sum(loss_sg) * g_lam
            val_loss += tloss.item()
            n_val += labels.size(dim=0)

    val_loss /= n_val
    accuracy = 100 * correct / n_val
    print(
        f"Epoch {epoch+1}, Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}%"
    )
    return accuracy


def RGB2BGR(im):
    assert im.mode == "RGB"
    r, g, b = im.split()
    return Image.merge("RGB", (b, g, r))


def load_checkpoint(model, checkpoint_file, from_SC=False, out_n_classes=None):
    print("Resuming from {}".format(checkpoint_file))
    # checkpoint = torch.load(checkpoint_file, map_location='cpu')
    torch.serialization.add_safe_globals([collections.defaultdict])
    torch.serialization.add_safe_globals([float])
    checkpoint = torch.load(checkpoint_file, map_location=torch.device("cuda"))
    if from_SC:
        # for key in ['model', 'optimizer', 'train_sampler', 'test_sampler', 'lr_scheduler', 'scaler', 'metrics_train', 'metrics_test', 'metrics_sys', 'best_accuracy']:
        #     print("key {} value {}".format(key, checkpoint[key]))
        saved_model = checkpoint["model"]
    else:
        saved_model = checkpoint

    msg = model.load_state_dict(saved_model, strict=True)

    if out_n_classes is not None:
        # Perform surgery on the model.  Remove the last layer and add new linear layer
        # note: last_layer = model.module.out = nn.Linear(in_features, out_features) when wrapped with DataParallel
        last_layer = model.module.out
        # print(last_layer)
        model.module.out = nn.Linear(last_layer.in_features, out_n_classes)
    # print(msg)
    # checkpoint = torch.load(checkpoint_file, map_location=torch.device('cuda'))
    # model.load_state_dict(torch.load(checkpoint)) #, strict=False)
    model.cuda(0)
    model.eval()


def main(args):
    # create a classifier model with number of classes as the output dimension.
    if args.pretrain_C is None:
        model = dbt.create_model(args.C, False)
    else:
        model = dbt.create_model(args.pretrain_C, False)

    model = model.cuda()
    model = torch.nn.DataParallel(model)

    if args.resume:
        load_checkpoint(model, args.resume, from_SC=args.SC, out_n_classes=args.C)

    # define loss function (criterion) and optimizer
    optimizer = torch.optim.Adam(
        [
            {"params": model.parameters(), "lr": args.modellr},
        ],
        eps=args.eps,
        weight_decay=args.weight_decay,
    )
    # Set LR scheduler
    scheduler = dbt.lr_scheduler(
        optimizer,
        args.modellr,
        args.lr_decay,
        args.lr_decay_epoch,
        args.epochs,
        name=args.lr_decay_type,
        lr_min=args.lr_min,
        t_mult=args.lr_t_mult,
    )

    cudnn.benchmark = True

    # load data
    traindir = os.path.join(args.data, "train")
    testdir = os.path.join(args.data, "test")

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    input_dim_resize = 1024
    input_dim_crop = args.image_size

    # TODO: investigate this?
    # For EfficientNet with advprop
    # normalize = transforms.Lambda(lambda img: img * 2.0 - 1.0)

    # criterion = dbt.dbt_loss_fcn(args.g_lambda)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing).cuda()
    criterion_validate = nn.CrossEntropyLoss().cuda()
    test_transforms = transforms.Compose(
        [
            transforms.Resize(input_dim_resize),
            transforms.CenterCrop(input_dim_crop),
            transforms.ToTensor(),
            normalize,
        ]
    )
    test_dataset = datasets.ImageFolder(testdir, test_transforms, allow_empty=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    # print("test class to idx {}".format(test_dataset.class_to_idx))

    if not args.eval_only:
        if args.rand_config:
            # note mean is 255 * (0.485, 0.456, 0.406).  TODO define
            # mean in one spot to make sure normalize and rand augment
            # have same mean.
            rand_tfm = rand_augment_transform(
                config_str=args.rand_config, hparams={"img_mean": (124, 116, 104)}
            )

            print("Using random augmentation...")
            train_dataset = datasets.ImageFolder(
                traindir,
                transforms.Compose(
                    [
                        RandomResizedCropAndInterpolation(input_dim_crop),
                        transforms.RandomHorizontalFlip(),
                        rand_tfm,
                        transforms.ToTensor(),
                        normalize,
                    ]
                ),
            )
        else:
            print("Not using random augmentation...")
            # note mean is 255 * (0.485, 0.456, 0.406).  TODO define
            train_dataset = datasets.ImageFolder(
                traindir,
                transforms.Compose(
                    [
                        transforms.RandomResizedCrop(input_dim_crop),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        normalize,
                    ]
                ),
            )

        if args.use_cutmix_or_mixup:
            cutmix = v2.CutMix(num_classes=args.C)
            mixup = v2.MixUp(num_classes=args.C)
            cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])
            def collate_fn(batch):
                return cutmix_or_mixup(*default_collate(batch))

        else:
            collate_fn = None

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )

        if train_dataset.class_to_idx != test_dataset.class_to_idx:
            raise Exception(
                "train class to index {} != test {}".format(
                    train_dataset.class_to_idx, test_dataset.class_to_idx
                )
            )

        print(
            "Training data in {} Cross-Validation data in {}".format(traindir, testdir)
        )
        best_recall = 0
        for epoch in range(args.start_epoch, args.epochs):
            print(
                "Training in Epoch[{}]. Current learning rate {}".format(
                    epoch, scheduler.get_last_lr()
                )
            )

            # train for one epoch
            train(
                train_loader,
                model,
                criterion,
                args.g_lambda,
                optimizer,
                args,
                num_avg_iter=args.loss_every,
            )

            # Run validation and advance LR by 1 step
            recall = validate(test_loader, model, criterion_validate, args.g_lambda, epoch, args)
            scheduler.step()

            last_epoch = epoch

            # Save the best model
            if recall > best_recall:
                best_recall = recall
                new_best_model = True
            else:
                new_best_model = False
            tag = "New Best Recall" if new_best_model else "Next Recall"
            print("{}@1: {recall:.3f}\n".format(tag, recall=recall))
            # print('Recall@1, 2, 4, 8: {recall[0]:.3f}, {recall[1]:.3f}, {recall[2]:.3f}, {recall[3]:.3f}; NMI: {nmi:.3f} \n'
            #       .format(recall=recall, nmi=nmi))
            if new_best_model:
                print("Saving new best model!")
                fn = "{}.pth".format(f"best_model_{epoch}")
                torch.save(model.state_dict(), fn)
                print("Model saved to", fn)
    else:
        print("Evaluation Mode...")
        last_epoch = args.epoch - 1

    # evaluate on validation set
    recall = validate(test_loader, model, criterion, args.g_lambda, last_epoch, args)
    # print('Recall@1, 2, 4, 8: {recall[0]:.3f}, {recall[1]:.3f}, {recall[2]:.3f}, {recall[3]:.3f}; NMI: {nmi:.3f} \n'
    #       .format(recall=recall, nmi=nmi))
    print("Last Recall@1: {recall:.3f}\n".format(recall=recall))

    # Save the model
    if not args.eval_only:
        fn = "{}.pth".format("last_model")
        print("Saving model!")
        torch.save(model.state_dict(), fn)
        print("Model saved to", fn)

    # Below test code reads back in the model and checks that
    # the answer is the same.  It is, so moving on to how the
    # model is saved.

    # load_checkpoint(model, fn)
    # nmi_1, recall_1, tio_1 = \
    # validate(test_loader, test_transforms, model, args)
    # print('Reload Model Recall@1, 2, 4, 8: {recall[0]:.3f}, {recall[1]:.3f}, {recall[2]:.3f}, {recall[3]:.3f}; NMI: {nmi:.3f} \n'
    # .format(recall=recall_1, nmi=nmi_1))
    # delta1 = tio_1 - tio
    # l2d_1 = torch.linalg.norm(delta1)
    # print("ld1 {}".format(l2d_1))


parser = argparse.ArgumentParser(description="PyTorch Training")
parser.add_argument("data", help="path to dataset")
parser.add_argument(
    "-j", "--workers", default=2, type=int, help="number of data loading workers"
)
parser.add_argument(
    "--epochs", default=500, type=int, help="number of total epochs to run"
)
parser.add_argument("--start-epoch", default=0, type=int, help="manual epoch number")
parser.add_argument("-b", "--batch-size", default=32, type=int, help="mini-batch size")
parser.add_argument(
    "--modellr", default=0.05, type=float, help="initial model learning rate"
)
parser.add_argument(
    "--lr-decay", default=0.1, type=float, help="learning rate decay factor."
)
parser.add_argument(
    "--lr-min",
    type=float,
    default=None,
    help="If specified, the lr decay rate parameters will be adjusted so that we reach the mininmum in the total number of epochs",
)
parser.add_argument(
    "--lr-decay-epoch",
    default=50,
    type=float,
    help="specification of epoch to decay lr",
)
parser.add_argument(
    "--lr-decay-type",
    type=str,
    default="step",
    choices=["none", "step", "exp", "cos", "cosd"],
    help="lr decay function type",
)
parser.add_argument(
    "--lr-t_mult",
    type=int,
    default=1,
    help="for cos decay type, multiple of period after decay epochs",
)
parser.add_argument(
    "--wd",
    "--weight-decay",
    default=1e-4,
    type=float,
    help="weight decay",
    dest="weight_decay",
)
parser.add_argument("--gpu", default=None, type=int, help="GPU id to use.")
parser.add_argument("--eps", default=0.01, type=float, help="epsilon for Adam")
parser.add_argument("--rate", default=0.1, type=float, help="decay rate")
parser.add_argument("--dim", default=64, type=int, help="dimensionality of embeddings")
parser.add_argument("--freeze-BN", action="store_true", help="freeze bn")
parser.add_argument(
    "--g-lambda", default=3.0e-04, type=float, help="weight of group bilinear loss"
)
parser.add_argument("-C", default=98, type=int, help="Number of classes")
parser.add_argument(
    "--pretrain-C", default=None, type=int, help="Number of classes from pre-training"
)
parser.add_argument(
    "--rand_config", default="rand-mstd1", help="Random augment configuration string"
)
parser.add_argument("--resume", default=None, help="resume from given file")
parser.add_argument(
    "--eval-only", default=False, action="store_true", help="evaluate model only"
)
parser.add_argument(
    "--image-size", default=598, type=int, help="Size the crops to this dimension"
)
parser.add_argument(
    "--loss-every",
    default=500,
    type=int,
    help="Report running average of training loss at this count of iterations",
)
parser.add_argument("--SC", action="store_true", help="Strong Compute checkpoint file")
parser.add_argument(
    "-M", "--use-cutmix-or-mixup", action="store_true", help="use CutMix or Mixup"
)
parser.add_argument("--label-smoothing", type=float, default=0.1)
if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
