from mxnet.gluon.loss import SigmoidBinaryCrossEntropyLoss
import mxnet.ndarray as nd
from mxnet import gluon, autograd
import mxnet as mx
import time
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from mxnet import init
from mxnet.gluon import nn

net = nn.Sequential()

net.add(nn.Dense(64,                    # Dense layer-1 with 64 units
                 #                  in_units=3,            # Input size of 3 is expected
                 activation='tanh'),    # Tanh activation is applied
        nn.Dropout(.4),                 # Apply random 40% drop-out to layer_1

        nn.Dense(64,                   # Dense layer-2 with 64 units
                 activation='tanh'),    # Tanh activation is applied

        nn.Dropout(.3),                 # Apply random 30% drop-out to layer_2

        nn.Dense(1))                    # Output layer with single unit

print(net)


net.initialize(init=init.Xavier())


X, y = make_circles(n_samples=750, shuffle=True,
                    random_state=42, noise=0.05, factor=0.3)


def plot_dataset(X, y, title):

    # Activate Seaborn visualization
    sns.set()

    # Plot both classes: Class1->Blue, Class2->Red
    plt.scatter(X[y == 1, 0], X[y == 1, 1], c='blue', label="class 1")
    plt.scatter(X[y == 0, 0], X[y == 0, 1], c='red', label="class 2")
    plt.legend(loc='upper right')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.title(title)
    plt.show()


plot_dataset(X, y, title="Dataset")


context = mx.cpu()       # Using CPU resource; mx.gpu() will use GPU resources if available
net = nn.Sequential()
net.add(nn.Dense(10, in_units=2, activation='relu'),
        nn.Dense(10, activation='relu'),
        nn.Dense(1, activation='sigmoid'))
net.initialize(init=init.Xavier(), ctx=context)

# Split the dataset into two parts: 80%-20% split
X_train, X_val = X[0:int(len(X)*0.8), :], X[int(len(X)*0.8):, :]
y_train, y_val = y[:int(len(X)*0.8)], y[int(len(X)*0.8):]

# Use Gluon DataLoaders to load the data in batches
batch_size = 4           # How many samples to use for each weight update
train_dataset = gluon.data.ArrayDataset(nd.array(X_train), nd.array(y_train))
train_loader = gluon.data.DataLoader(train_dataset, batch_size=batch_size)

# Move validation dataset in CPU/GPU context
X_val = nd.array(X_val).as_in_context(context)
y_val = nd.array(y_val).as_in_context(context)

epochs = 50              # Total number of iterations
learning_rate = 0.01     # Learning rate

# Define the loss. As we used sigmoid in the last layer, use from_sigmoid=True
binary_cross_loss = SigmoidBinaryCrossEntropyLoss(from_sigmoid=True)

# Define the trainer, SGD with learning rate
trainer = gluon.Trainer(net.collect_params(),
                        'sgd',
                        {'learning_rate': learning_rate}
                        )


train_losses = []
val_losses = []
for epoch in range(epochs):
    start = time.time()
    training_loss = 0
    # Build a training loop, to train the network
    for idx, (data, target) in enumerate(train_loader):

        data = data.as_in_context(context)
        target = target.as_in_context(context)

        with autograd.record():
            output = net(data)
            L = binary_cross_loss(output, target)
            training_loss += nd.sum(L).asscalar()
            L.backward()
        trainer.step(data.shape[0])

    # Get validation predictions
    val_predictions = net(X_val)
    # Calculate the validation loss
    val_loss = nd.sum(binary_cross_loss(val_predictions, y_val)).asscalar()

    # Take the average losses
    training_loss = training_loss / len(y_train)
    val_loss = val_loss / len(y_val)

    train_losses.append(training_loss)
    val_losses.append(val_loss)

    end = time.time()
    # Print the losses every 10 epochs
    if (epoch == 0) or ((epoch+1) % 10 == 0):
        print("Epoch %s. Train_loss %f Validation_loss %f Seconds %f" %
              (epoch, training_loss, val_loss, end-start))


plt.plot(train_losses, label="Training Loss")
plt.plot(val_losses, label="Validation Loss")
plt.title("Loss values")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()
