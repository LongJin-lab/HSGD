import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from   scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import os
from network import Net,block

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
planes        = 12
num_epochs    = 10
learning_rate = 0.001
train_num     = 20000
num_epochs    = 500

#定义函数计算移动平均损失值
def moving_average(a, w=10):
    if len(a) < w:
        return a[:]
    return [val if idx < w else sum(a[(idx-w):idx])/w for idx, val in enumerate(a)]
 
def plot_losses(losses):
    avgloss= moving_average(losses) #获得损失值的移动平均值
    plt.figure(1)
    plt.subplot(211)
    plt.plot(range(len(avgloss)), avgloss, 'b--')
    plt.xlabel('step number')
    plt.ylabel('Training loss')
    plt.title('step number vs. Training loss')
    plt.show()

#1.1 载入样本
titanic_data = pd.read_csv('titanic3.csv')
print(titanic_data.columns)

titanic_data = pd.concat(
    [titanic_data,
     pd.get_dummies(titanic_data['sex']),
     pd.get_dummies(titanic_data['embarked'],prefix="embark"),
     pd.get_dummies(titanic_data['pclass'],prefix="class")],axis=1
)
titanic_data["age"]  = titanic_data["age"].fillna(titanic_data["age"].mean()) # 乘客年龄
titanic_data["fare"] = titanic_data["fare"].fillna(titanic_data["fare"].mean()) # 乘客票
titanic_data = titanic_data.drop(['name','ticket','cabin','boat','body','home.dest','sex','embarked','pclass'], axis=1)

 
labels = titanic_data["survived"].to_numpy()
 
titanic_data = titanic_data.drop(['survived'],axis=1)
data = titanic_data.to_numpy()
 
feature_names = list(titanic_data.columns)

 
train_indices = np.random.choice(len(labels),int(0.8 * len(labels)),replace = False)
test_indices = list(set(range(len(labels))) - set(train_indices)) #将剩余部分设为测试集
 
train_features = data[train_indices]
train_labels   = labels[train_indices]
 
test_features  = data[test_indices]
test_labels    = labels[test_indices]



torch.manual_seed(0) # 设置随机种子函数
if __name__ == '__main__':
    
    model = Net(block,planes)
    optimizer = torch.optim.SGD(model.parameters(),lr = 0.1)
    critirion = nn.L1Loss()

    # 将输入的样本标签转化为标量
    input_tensor = torch.from_numpy(train_features).type(torch.FloatTensor)
    label_tensor = torch.from_numpy(train_labels)
    losses = [] 
    
    for epoch in range(num_epochs): 
        print(1+model.c,-2*model.c,model.c) 
        scores = model(input_tensor)
        loss = critirion(scores.squeeze(dim=1),label_tensor)
        losses.append(loss.item())
        optimizer.zero_grad()    
        loss.backward()          
        optimizer.step()        
        
        print('Epoch {}/{} => Loss: {:.2f}'.format(epoch + 1, num_epochs, loss.item()))

    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/titanic_model.pt')
    plot_losses(losses)
 
    # 输出训练结果
    # tensor.detach():从计算图中脱离出来，返回一个新的tensor，新的tensor和原tensor共享数据内存，（这也就意味着修改一个tensor的值，另外一个也会改变），
    #                  但是不涉及梯度计算。在从tensor转换成为numpy的时候，如果转换前面的tensor在计算图里面（requires_grad = True），那么这个时候只能先进行detach操作才能转换成为numpy
    out_probs = model(input_tensor).detach().numpy()
    out_classes = np.argmax(out_probs, axis=1)
    print("Train Accuracy:", sum(out_classes == train_labels) / len(train_labels))
 
    # 测试模型
    test_input_tensor = torch.from_numpy(test_features).type(torch.FloatTensor)
    out_probs = model(test_input_tensor).detach().numpy()
    out_classes = np.argmax(out_probs, axis=1)
    print("Test Accuracy:", sum(out_classes == test_labels) / len(test_labels))
  
