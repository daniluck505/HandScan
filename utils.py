import yaml
import os
from datetime import datetime
import torch
from PIL import Image
from tqdm import tqdm
import pickle



def load_options(path):
    with open(path, 'r') as f:
        options = yaml.safe_load(f)
    return options

def download_kaggle_data(name, json=True):
    if json and ('kaggle.json' not in os.listdir()):
        raise Exception(f'kaggle.json not found')
    os.system('pip install kaggle')
    os.system('mkdir -p ~/.kaggle')
    os.system('cp kaggle.json ~/.kaggle/')
    os.system('chmod 600 ~/.kaggle/kaggle.json')
    os.system(f'kaggle datasets download "{name}"')
    os.system(f'unzip "{name.split("/")[-1]}.zip"')
    
    
def save_results(options, model, loss_history, val_history, epochs):
    if not os.path.exists(f'{options["name"]}'):
        os.system(f'mkdir {options["name"]}')
    now = datetime.now()
    now = str(now).split('.')[0].replace(' ','_' )
    name = f'{options["network"]["arch"]}_epoch_{epochs}_{now}'
    torch.save(model.state_dict(), f'{options["name"]}/{name}_weights.pt') 

    with open(F'{options["name"]}/{name}_results.pkl', 'wb') as file: 
        dict_save = {'loss_history': loss_history,
                     'val_history': val_history,
                     'options': options}
        pickle.dump(dict_save, file) 


def load_results(file_name):
    with open(file_name, 'rb') as file: 
        dict_load = pickle.load(file) 
    return dict_load


def load_model_weights(model, weights):
    model.load_state_dict(torch.load(weights, weights_only=True))
    return model

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class TextProcessor:
    def __init__(self, chars):
        char = ['^'] + sorted(chars)
        self.vocab_size = len(char)
        self.int2char = dict(enumerate(char))
        self.char2int = {char: ind for ind, char in self.int2char.items()}

    def get_vocab_size(self):
        return self.vocab_size

    def encode(self, texts):
        ''' text -> tensor '''
        text_length = []
        encoded_texts = []
        for t in texts:
            text_length.append(len(t))
            for c in t.lower():
                if c in self.char2int:
                    encoded_texts.append(self.char2int[c])

        return torch.tensor(encoded_texts), torch.tensor(text_length)

    def decode(self, encoded_text):
        ''' tensor -> text'''
        text = []
        for i in encoded_text:
            text.append(self.int2char.get(i.item()))
        return self.remove_duplicate(text)

    def remove_duplicate(self, text):
        decoded_text = ''
        for i, t in enumerate(text):
            if (t == '^') or (i > 0 and t == text[i-1]):
                continue
            decoded_text += t
        return decoded_text
    

def fix_png_files(folder_path):
    for filename in tqdm(os.listdir(folder_path)):
        file_path = os.path.join(folder_path, filename)
        
        if filename.lower().endswith('.png'):
            try:
                with Image.open(file_path) as img:
                    img = img.convert("RGBA")
                    img.save(file_path, "PNG")
            except Exception as e:
                print(f"Файл {filename} невозможно исправить. Удаление: {e}")
                os.remove(file_path)


def remove_invalid_png_files(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        if filename.lower().endswith('.png'):
            try:
                with Image.open(file_path) as img:
                    img.verify()  
            except Exception as e:
                print(f"Удаление файла {filename} из-за ошибки: {e}")
                os.remove(file_path)


def merge_dicts(old_dict, new_dict):
    for key, value in new_dict.items():
        if key in old_dict:
            if isinstance(old_dict[key], dict) and isinstance(value, dict):
                merge_dicts(old_dict[key], value)
            else:
                old_dict[key] = value
        else:
            old_dict[key] = value
    return old_dict
    