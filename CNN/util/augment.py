import torch
import numpy as np
from .datasets import *
import torchvision
import itertools
import matplotlib.pyplot as plt
from PIL import Image


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()  # 返回[0,batch_size-1]的随机排序
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def cutout(mask_size, p, cutout_inside):
    """
    默认用黑色填充
    mask_size: 被裁剪的大小
    p: 应用裁剪的概率
    cutout_inside:裁剪是否在图像内部发生
    return: 一个裁剪函数, 应用在image上可以返回裁剪后的函数
    """
    mask_size_half = mask_size // 2
    offset = 1 if mask_size % 2 == 0 else 0

    def _cutout(batch_image):
        batch_image = torch.clone(batch_image)
        batch_size = batch_image.shape[0]
        rand = torch.rand(batch_size, 1)
        is_cutout = rand < p

        h, w = batch_image.shape[2:]

        if cutout_inside:
            cxmin, cxmax = mask_size_half, w + offset - mask_size_half
            cymin, cymax = mask_size_half, h + offset - mask_size_half
        else:
            cxmin, cxmax = 0, w + offset
            cymin, cymax = 0, h + offset
        # (cx,cy)裁剪中心
        cx = torch.randint(cxmin, cxmax, size=(batch_size, 1))
        cy = torch.randint(cymin, cymax, size=(batch_size, 1))
        xmin = cx - mask_size_half
        ymin = cy - mask_size_half
        xmax = xmin + mask_size
        ymax = ymin + mask_size
        xmin = torch.clamp(xmin, min=0)
        ymin = torch.clamp(ymin, min=0)  # 用于替换小于或大于某个值
        xmax = torch.clamp(xmax, max=h)
        ymax = torch.clamp(ymax, max=h)
        for idx, flag in enumerate(is_cutout.squeeze()):
            if flag == 0:
                continue
            # Create mask for specific image
            full_mask = torch.zeros(
                (3, ymax[idx]-ymin[idx], xmax[idx]-xmin[idx]))
            batch_image[idx, :, ymin[idx]:ymax[idx],
                        xmin[idx]:xmax[idx]] = full_mask
        return batch_image

    return _cutout


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    # np.clip限制长度和高度在一定范围内
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def cutmix(beta, cutmix_prob, inputs, targets):
    r = np.random.rand(1)
    lam = 1
    if beta > 0 and r < cutmix_prob:
        # generate mixed sample
        lam = np.random.beta(beta, beta)
        rand_index = torch.randperm(inputs.size()[0]).cuda()
        target_a = targets
        target_b = targets[rand_index]
        bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam)
        inputs[:, :, bbx1:bbx2, bby1:bby2] = inputs[rand_index,
                                                    :, bbx1:bbx2, bby1:bby2]
        # adjust lambda to exactly match pixel ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) /
                   (inputs.size()[-1] * inputs.size()[-2]))
    else:
        target_a,target_b=targets,targets
    return inputs, target_a, target_b, lam


def generate_image(image_tensor, augment, id):
    # 将图像张量转换为 PIL 图像
    image = torchvision.transforms.functional.to_pil_image(image_tensor)
    # 保存图像
    image.save("HW2/image/"+augment+"/image_"+str(id)+".jpg")


if __name__ == '__main__':
    batch_size = 3
    train_dataloader, _ = load_dataset(batch_size)
    for data in train_dataloader:
        inputs, targets = data
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        inputs = inputs.to(device)
        targets = targets.to(device)
        print(inputs[1])
        for b in range(batch_size):
            generate_image(inputs[b, :, :, :], "baseline", b+1)
        inputs_mixup, _, _, _ = mixup_data(inputs, targets, 1.)
        for b in range(batch_size):
            generate_image(inputs_mixup[b, :, :, :], "mixup", b+1)
        inputs_cutmix, _, _, _ = cutmix(1, 1, inputs, targets)
        for b in range(batch_size):
            generate_image(inputs_cutmix[b, :, :, :], "cutmix", b+1)
        Cutout = cutout(8, 1, False)
        inputs_cutout = Cutout(inputs)
        for b in range(batch_size):
            generate_image(inputs_cutout[b, :, :, :], "cutout", b+1)
        break
    augment_method = ["baseline", 'mixup', 'cutout', 'cutmix']
    idx = ['1', '2', '3']
    image_paths = ['HW2/image/' + method + "/image_"+ind +
                   ".jpg" for method, ind in itertools.product(augment_method, idx)]
    fig, axes = plt.subplots(4, 3, figsize=(10, 8))
    for i, image_path in enumerate(image_paths):
        # 打开图像文件
        image = Image.open(image_path)

        # 计算子图位置
        row = i // 3
        col = i % 3

        # 在子图网格中显示图像
        axes[row, col].imshow(image)
        axes[row, col].axis("off")

        if col == 0:
            axes[row, col].set_title(augment_method[row])
    plt.tight_layout()
    plt.savefig("HW2/image/image_grid.png")
    plt.show()
