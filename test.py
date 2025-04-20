import datetime
import matplotlib.pyplot as plt



def lr_scheduler(epoch, lr, decay=1):
    lr = lr/(1+(decay*epoch))
    return lr

lrs = []
lr = 0.01
decay = 2
for epoch in range(10):
    lrs.append(lr_scheduler(epoch, lr, decay))

plt.plot(lrs, marker='.')
# plt.show()

print(datetime.datetime.now())


