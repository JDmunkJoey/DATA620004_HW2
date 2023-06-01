import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.utils.data.dataloader as dataloader


# Root指数据集所在的根目录
# train指从训练集还是测试集创建
def explore_data(dic):
    """
    查看dataset中的数据结构
    :param dic:
    :return:
    """
    print(dic.keys(
    ))  # [b'filenames', b'batch_label', b'fine_labels', b'coarse_labels', b'data']
    print(f"文件名称: {dic[b'filenames'][0]}")
    print(f'total image numbers: {len(dic[b"filenames"])}')  # 总共多少张训练图片
    # 细粒度分类有多少类别 100
    print(
        f"细粒度label range :{min(dic[b'fine_labels'])}, {max(dic[b'fine_labels'])}")
    # 粗粒度分类有多少类别 20
    print(
        f"粗粒度label range :{min(dic[b'coarse_labels'])}, {max(dic[b'coarse_labels'])}")
    print(f"data的类型：{type(dic[b'data'])}")
    print(f"data的shape: {dic[b'data'].shape}")
    print(len(dic[b'data'][0]))


def unpickle(file):
    """
    将二进制数据文件解码
    :param file: 二进制文件路径
    :return:
    """
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def load_dataset(batch_size):
    train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]))
    train_dataloader = dataloader.DataLoader(train_dataset, batch_size=batch_size,
                                             shuffle=True, num_workers=8)
    test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]))
    test_dataloader = dataloader.DataLoader(test_dataset, batch_size=batch_size,
                                            shuffle=False, num_workers=8)
    return train_dataloader, test_dataloader


# test_dataset = datasets.CIFAR100(root='./tes')
if __name__ == '__main__':
    train_dataloader, _ = load_dataset(1)
    for i, data in enumerate(train_dataloader):
        print(data[0])
        if i > 10:
            break
    D = unpickle('./data/cifar-100-python/test')
    explore_data(D)
