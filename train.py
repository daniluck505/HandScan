import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
import time

from dataset import *
from utils import *
from model import *
from trainer import *

# load config file
conf = load_options('config.yml')

# make data loader
dconf = conf['dataset']
set_chars = set(dconf['set_chars'])

transforms = tv.transforms.Compose([
    tv.transforms.ToPILImage(),
    tv.transforms.ColorJitter(brightness=.5, hue=.5),
    tv.transforms.RandomRotation(10, fill=255),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
ds_train = DatasetText(dconf['train_path'], 
                       dconf['train_labels'], 
                       transforms, set_chars, 
                       dconf['max_len'])

transforms = tv.transforms.Compose([
    tv.transforms.ToTensor(),
    tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
ds_test = DatasetText(dconf['test_path'], 
                       dconf['test_labels'], 
                       transforms, set_chars, 
                       dconf['max_len'])

train_loader = torch.utils.data.DataLoader(
    ds_train, shuffle=dconf['shuffle'],
    batch_size=dconf['batch_size'], num_workers=0, drop_last=True)

test_loader = torch.utils.data.DataLoader(
    ds_test, shuffle=False,
    batch_size=dconf['batch_size'], num_workers=0, drop_last=False)

# prepare model
nconf = conf['network']

chars = sorted(list(set_chars))
tp = TextProcessor(chars)

vocab_size = tp.get_vocab_size()
model = get_model(nconf['arch'], vocab_size)
print('Count parameters: ', count_parameters(model))
if nconf['weights'] is not None:
    model = load_model_weights(model, nconf['weights'])
model = model.to(nconf['device'])

optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True)
trainer = Trainer(model, optimizer, calculate_loss, tp, train_loader, test_loader, 5)
if len(trainer.train_loss) == 0:
    load_opt = load_results(nconf['options'])
    trainer.train_loss = load_opt['loss_history']
    trainer.val_loss = load_opt['val_history']
    trainer.epoch_sum = int(nconf['weights'].split('_')[-4])
    # conf = merge_dicts(conf, load_opt['options'])
    # conf = load_opt['options']

# train
tconf = conf['train']

if tconf['epochs'] != 0:
    trainer.train_model(tconf)
    if tconf['plot']:
        trainer.plot_history((6, 2))

# Validate
if tconf['see_test_predict']:
    loss = 0
    all_time = 0
    count = 0
    for i, batch in enumerate(test_loader):
        print(f'Batch {i}')
        start_time = time.time()

        loss += trainer.batch_validate(batch, tconf['device'], False)

        end_time = time.time()
        all_time += end_time - start_time
        print(len(batch['img']))
        count += len(batch['img'])

    mean_time = all_time / count
    mean_loss = loss / (i+1)
    print(f'Mean time: {mean_time:.2f} second')
    print(f'Data mean WER: {mean_loss:.2f}')

save_flag = input('Save results? (y/n) ')
if save_flag != 'n':
    save_results(conf, model, trainer.train_loss, trainer.val_loss, trainer.epoch_sum)

print('\n\nFinish')