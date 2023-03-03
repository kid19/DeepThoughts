import os

import argparse

import torch

import torchvision

from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from vit_model import vit_base_patch16_224_in21k as create_model
from utils import evaluate


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose([transforms.Resize([224, 224]),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    test_dataset = torchvision.datasets.ImageFolder('./data/large_covid_data_test', transform=data_transform)


    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw)

    model = create_model(num_classes=2, has_logits=False).to(device)
    weights_dict = torch.load(args.weights, map_location=device)
    model.load_state_dict(weights_dict, strict=True)

    for epoch in range(args.epochs):

        # test
        evaluate(model=model, data_loader=test_loader, device=device, epoch=epoch)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=8)

    parser.add_argument('--model-name', default='', help='create model name')

    # 预训练权重路径，如果不想载入就设置为空字符
    parser.add_argument('--weights', type=str, default='./weights/model-28.pth',
                        help='initial weights path')
    # 是否冻结权重
    parser.add_argument('--freeze-layers', type=bool, default=True)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)