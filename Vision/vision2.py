import os
from d2l import mxnet as d2l
import mxnet as mx
from mxnet import gluon, init, nd, autograd
from mxnet.gluon import nn
from mxnet.gluon.data.vision import transforms
from mxnet.gluon.loss import SoftmaxCrossEntropyLoss
import numpy as np

transform_train = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize()
])

transform_test = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize()
])

batch_size = 16

path = '../data/minc-2500'
train_path = os.path.join(path, 'train')
val_path = os.path.join(path, 'val')
test_path = os.path.join(path, 'test')

train_loader = gluon.data.DataLoader(
    gluon.data.vision.ImageFolderDataset(
        train_path).transform_first(transform_train),
    batch_size=batch_size, shuffle=True)

validation_loader = gluon.data.DataLoader(
    gluon.data.vision.ImageFolderDataset(
        val_path).transform_first(transform_test),
    batch_size=batch_size, shuffle=False)

test_loader = gluon.data.DataLoader(
    gluon.data.vision.ImageFolderDataset(
        test_path).transform_first(transform_test),
    batch_size=batch_size, shuffle=False)


def FineTuneAlexnet(classes, ctx):
    '''
    classes: number of the output classes 
    ctx: training context (CPU or GPU)
    '''
    finetune_net = gluon.model_zoo.vision.alexnet(
        classes=classes, pretrained=False, ctx=ctx)
    finetune_net.initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)
    pretrained_net = gluon.model_zoo.vision.alexnet(pretrained=True, ctx=ctx)
    finetune_net.features = pretrained_net.features

    return finetune_net


ctx = d2l.try_gpu()  # Create neural net on CPU or GPU depending on your training instances
num_outputs = 6  # 6 output classes
net = FineTuneAlexnet(num_outputs, ctx)
net


learning_rate = 0.001
trainer = gluon.Trainer(net.collect_params(), 'sgd', {
                        'learning_rate': learning_rate})

softmax_cross_etropy_loss = gluon.loss.SoftmaxCrossEntropyLoss()


def finetune_accuracy(output, label):
    # output: (batch, num_output) float32 ndarray
    # label: (batch, ) int32 ndarray
    return (output.argmax(axis=1) == label.astype('float32')).mean()


epochs = 10

for epoch in range(epochs):

    train_loss, val_loss, train_acc, valid_acc = 0., 0., 0., 0.

    # Training loop: (with autograd and trainer steps, etc.)
    # This loop does the training of the neural network (weights are updated)
    for i, (data, label) in enumerate(train_loader):
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        with autograd.record():
            output = net(data)
            loss = softmax_cross_etropy_loss(output, label)
        loss.backward()
        train_acc += finetune_accuracy(output, label)
        train_loss += loss.mean()
        trainer.step(data.shape[0])

    # Validation loop:
    # This loop tests the trained network on validation dataset
    # No weight updates here
    for i, (data, label) in enumerate(validation_loader):
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        output = net(data)
        valid_acc += finetune_accuracy(output, label)
        val_loss += softmax_cross_etropy_loss(output, label).mean()

    # Take averages
    train_loss /= len(train_loader)
    train_acc /= len(train_loader)
    val_loss /= len(validation_loader)
    valid_acc /= len(validation_loader)

    print("Epoch %d: train loss %.3f, train acc %.3f, val loss %.3f, val acc %.3f" % (
        epoch, train_loss.asnumpy()[0], train_acc.asnumpy()[0], val_loss.asnumpy()[0], valid_acc.asnumpy()[0]))

net.save_parameters("my_model")
