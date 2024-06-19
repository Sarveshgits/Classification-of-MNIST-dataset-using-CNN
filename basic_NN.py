import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class Model(nn.Module):
    def __init__(self, in_features=4, h1=8,h2=6, out_features=3 ):
        super().__init__()
        self.f1=nn.Linear(in_features,h1)
        self.f2=nn.Linear(h1,h2)
        self.out=nn.Linear(h2,out_features)

    def forward(self, x):
        x = F.relu(self.f1(x))
        x = F.relu(self.f2(x))
        x=self.out(x)

        return x
    
model=Model()
    
url='https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv'
my_df=pd.read_csv(url)

my_df['variety']=my_df['variety'].replace('Setosa', 0.0)
my_df['variety']=my_df['variety'].replace('Versicolor', 1.0)
my_df['variety']=my_df['variety'].replace('Virginica', 2.0) 

X=my_df.drop('variety', axis =1)
y=my_df['variety']

X=X.values
y=y.values

X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2, random_state=41)

X_train=torch.FloatTensor(X_train)
X_test=torch.FloatTensor(X_test)
y_train=torch.LongTensor(y_train)
y_test=torch.LongTensor(y_test)


criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(), lr=0.01)


epochs=100
losses=[]
for i in range(epochs):
    y_pred = model.forward(X_train)

    loss=criterion(y_pred, y_train)
    losses.append(loss.detach().numpy())

    if i % 10==0:
        print(f'Epoch: {i} and loss: {loss}')

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


with torch.no_grad():
    y_eval=model.forward(X_test)
    loss=criterion(y_eval,y_test)

#Comparing loss of training data and testing data
print(loss)

#Testing our model with test data
correct=0
with torch.no_grad():
    for i, data in enumerate(X_test):
        y_val=model.forward(data)

        print(f'{i+1}.) {str(y_val)} \t {y_test[i]} \t{y_val.argmax().item()}')

        if y_val.argmax().item()==y_test[i]:
            correct+=1
print(f'We got {correct} correct!')

#New data-point
new_Iris= torch.tensor([6.8, 3.2, 5.9, 2.3])
with torch.no_grad():
    print(model(new_Iris))