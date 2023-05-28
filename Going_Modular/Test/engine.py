from tqdm.auto import tqdm
import torch
import os

def train(device,epochs,model,dataloader,loss_fn,optimizer):

    
    
    model.train()
    for epoch in range(epochs):

        train_acc = 0
        train_loss = 0

        for i,(x,y) in enumerate(dataloader):

            x,y = x.to(device),y.to(device)

            y_pred = model(x)

            loss = loss_fn(y_pred,y)

            train_loss += loss.item()

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            y_pred_class = torch.argmax(torch.softmax(y_pred,dim = -1),dim=-1)

            train_acc += torch.eq(y_pred_class,y).sum().item()/len(y_pred)

        train_acc /= len(dataloader)
        train_loss /= len(dataloader)

        if (epoch % 10 == 0):

            print(epoch)
            print(f"Train_ Loss:{train_loss},Train_acc:{train_acc}")


    return model 

def test(device,model,dataloader,loss_fn,optimizer):
    
    
    model.eval()
    with torch.inference_mode():

        loss = 0
        acc = 0
        
        for x,y in dataloader:

            x,y = x.to(device),y.to(device)

            y_pred = model(x)

            loss += loss_fn(y_pred,y)

            y_pred_class = torch.argmax(torch.softmax(y_pred,dim = -1),dim=-1)

            acc += torch.eq(y_pred_class,y).sum().item()/len(y_pred_class)

        loss /= len(dataloader)
        acc /= len(dataloader)

        print(f"loss : {loss} acc : {acc}")

    # torch.save(model.state_dict(),"/content/drive/MyDrive/Pytorch/Test")

    return model












    











