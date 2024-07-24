import torch

from src.train.dataset import get_dataset
from torch import nn

dataset_dict = get_dataset(csv_path='src/dataset/twitter_training.csv',
                           csv_column=['source_id', 'source', 'mood', 'context'],
                           eat_column_name='context',
                           label_column_name='mood')

train_set = dataset_dict['train'][0]
train_label = dataset_dict['train'][1]

test_set = dataset_dict['test'][0]
test_label = dataset_dict['test'][1]

valid_set = dataset_dict['valid'][0]
valid_label = dataset_dict['valid'][1]


device = "cuda" if torch.cuda.is_available() else "cpu"

print(f'device: {device}')


# Создаем модель
class DeadInsideFinder(nn.Modle):
    
    def __init__(self,):
        super().__init__()
        self.input_layer = nn.Linear(in_features=1, out_features=596)
        self.output_layer = nn.Linear(in_features=596, out_features=1)