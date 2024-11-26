import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
from tqdm import tqdm
from time import sleep
from datetime import datetime
import matplotlib.pyplot as plt
from jiwer import wer


class Trainer():
    def __init__(self, model, optimizer, loss_function, 
                 text_processor, train_loader, test_loader, rest_time=False):
        self.model = model
        self.optim = optimizer
        self.loss_function = loss_function
        self.tp = text_processor
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.rest_time = rest_time
        self.epoch_sum = 0
        self.train_loss, self.val_loss = [], []
    
    def train_model(self, options):
        for epoch in range(options['epochs']):
            train_loss = self.train_one_epoch(options, self.epoch_sum)
            loss = train_loss/len(self.train_loader)
            self.train_loss.append(loss)
            print(f'Train loss: {loss:.3f}')
            
            if options['validate']:
                val_loss = self.validate(options)
                loss = val_loss/len(self.test_loader)
                self.val_loss.append(loss)
                print(f'Val loss: {loss:.3f}')

            if self.rest_time:
                sleep(self.rest_time)
            
            print('\n')
            self.epoch_sum += 1

    
    def train_one_epoch(self, options, epoch):
        self.model.train()
        train_loss = 0
        for batch in (pbar := tqdm(self.train_loader)):
            img, label = batch['img'], batch['label']
            img = img.to(options['device'])

            self.optim.zero_grad()
            output = self.model(img)
            loss = self.loss_function(output, label, self.tp, options['device'])
            loss_item = loss.item()
            train_loss += loss_item
            loss.backward()
            self.optim.step()
            pbar.set_description(f'Epoch: {epoch}\tloss: {loss_item:.3f}')
        return train_loss
    
    def validate(self, options):
        self.model.eval()
        val_loss = 0
        for batch in self.test_loader:
            img, label = batch['img'], batch['label']
            img = img.to(options['device'])

            with torch.no_grad():
                output = self.model(img)
                loss = self.loss_function(output, label, self.tp, options['device'])
                loss_item = loss.item()
                val_loss += loss_item
            
        return val_loss
    

    def batch_validate(self, batch, device, see_img=False):
        labels = batch['label']
        self.model.eval()
        with torch.no_grad():
            output = self.model(batch['img'].to(device))
            self.loss_function(output, labels, self.tp, device)
        
        acc = 0
        all_true = ''
        all_predict = ''

        for ind in range(len(batch['img'])):
            if see_img:
                img_transform = batch['img'][ind].permute(1, 2, 0).numpy()
                plt.figure(figsize=(2, 4))
                plt.imshow(img_transform)
                plt.show();
            true = labels[ind]
            predict = self.tp.decode(output.argmax(2).permute(1, 0)[ind])

            all_true += true + ' '
            all_predict += predict + ' '

            if predict == true:
                acc += 1

            print(f'{ind}: {true} - {predict}')

        wer_loss = wer(all_true, all_predict)
        print('Accuracy:', acc)
        print(f'WER: {wer_loss:.2f}')
        return wer_loss
        

    
    def plot_history(self, size):
        plt.figure(figsize=size)
        lenght = list(range(len(self.train_loss)))
        plt.plot(lenght, self.train_loss)
        lenght = list(range(len(self.val_loss)))
        plt.plot(lenght, self.val_loss)
        plt.legend()
        now = datetime.now()
        now = str(now).split('.')[0].replace(' ','_' )
        plt.savefig(f'{now}_loss.png')
        plt.show();
        


def calculate_loss(inputs, texts, label_converter, device):
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)

    inputs = inputs.log_softmax(2)
    input_size, batch_size, _ = inputs.size()
    input_size = torch.full(size=(batch_size,), fill_value=input_size, dtype=torch.int32)

    encoded_texts, text_lens = label_converter.encode(texts)
    loss = criterion(inputs, encoded_texts.to(device), input_size.to(device), text_lens.to(device))

    return loss



def make_optimizer(options, model_params):
    params = options['optimizer']
    if params['name'] == 'adam':
        optimizer = torch.optim.Adam(model_params, lr=params['lr'], betas=(params['beta1'], params['beta2']))
    else:
        raise NotImplementedError(f'optimizer {params["optimizer"]} is not implemented')
    return optimizer
