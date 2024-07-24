import pandas

from sklearn.model_selection import train_test_split

# Получение информации о Фрейме 
def dataframe_info(data_frame:pandas.core.frame.DataFrame) -> None:

    print('--->Информация по DF<---\n')
    print(f'>shape:\n{data_frame.shape}\n')
    print(f'>columns:\n{data_frame.columns}\n')
    print(f'>index:\n{data_frame.index}\n')
    print(f'>dtypes:\n{data_frame.dtypes}\n')
    
    print('Первые 3:\n', data_frame[0:3])

# Получение датасета
def get_dataset(csv_path:str,
                csv_column:list,
                eat_column_name:str,
                label_column_name:str,
                get_info:bool=False
                ) -> dict:
    
    data_frame = pandas.read_csv(csv_path, delimiter=',')
    data_frame.columns = csv_column
    
    if get_info == True:
        dataframe_info(data_frame)

    eat = data_frame[eat_column_name]
    label = data_frame[label_column_name]

    # 60 20 20
    train_set, test_set, train_label, test_label = train_test_split(eat, label, test_size=0.4, random_state=42)
    test_set, valid_set, test_label, valid_label = train_test_split(test_set, test_label, test_size=0.5, random_state=42)
    
    dataset_dict = {"train":[train_set, train_label],
                    "test":[test_set, test_label],
                    "valid":[valid_set, valid_label]}
    
    return dataset_dict

if __name__ == '__main__':
    
    dataset = get_dataset(csv_path='src/dataset/twitter_training.csv',
                          csv_column=['source_id', 'source', 'mood', 'context'],
                          eat_column_name='context',
                          label_column_name='mood')
    
    print('\n\n',len(dataset['train'][0]), 
          len(dataset['test'][0]), 
          len(dataset['valid'][0]))