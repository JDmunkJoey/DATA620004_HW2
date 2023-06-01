此pj为使用AlexNet在CIFAR-100数据集中进行图像分类，并采用mixup、cutout和cutmix方法。

### 项目框架

#### checkpionts

保存的参数文件.pth，包含了模型的参数字典以及优化器的参数字典，每50个epoch保存一次

#### baseline_log、mixup_log、cutout_log、cutmix_log

用于tensorboard可视化的event文件，包含loss和prec准确率

终端输入：tensorboard --logdir=./baseline_log --port 8123，浏览器输入http://localhost:8123/即可查看。

#### image

baseline：挑选的三张图片

mixup：mixup后的三张图片

cutout：cutout后的三张图片

cutmix：cutmix后的三张图片

image_grid.jpg 12张图片拼接在一起的效果

#### data

存放CIFAR-100数据集

#### util

augment.py 实现mixup、cutout、cutmix的函数。实现将tensor转换为图片输出，选择了3张图片应用三种增强方法输出。

datasets.py 如何下载CIFAR-100数据集

eval.py 评价函数，topk准确率函数

misc.py 用于计算存储平均loss、prec等数值的类

#### log

cutmixlogging.log, cutoutlogging.log, mixuplogging .log

在训练输出的一些日志信息，包括每个epoch的准确率以及目前的最佳准确率，每个epoch记录花费的时间，loss，top1准确率和top5准确率。

#### model.py

AlexNet类别函数

#### nohup.out

由于是在后台运行的，在控制台的输出在这个文件中

#### train.py

训练函数

使用：(nohup) python train.py --params value

params：

- batch_size，每次训练的大小，默认128
- lr：初始学习率，默认0.001
- epochs：默认200
- phase：训练模式还是评估模式，默认训练
- model_pth：评估模式需要提供模型
- augment：选择数据增强的方法，只能输入：mixup或cutout或cutmix或baseline，baseline是指什么都不用
- alpha：mixup或cutmix的alpha参数，用于确定$\lambda$的Beta分布参数
- p：cutmix或cutout被裁剪的概率
- maskout_size：cutout裁剪的大小

训练过程：

1. 输入python train.py参数均采取默认参数。每50个epoch保存一次模型和优化器参数
2. 分别使用mixup、cutout、cutmix进行训练。终端输入python train.py --augment mixup
