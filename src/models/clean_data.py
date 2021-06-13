
from dotenv import find_dotenv, load_dotenv
from path import Path
import click
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import AutoModel, BertTokenizerFast
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler



# @click.command()
# @click.argument('df_name', type=click.Path(exists=True))
def clean_data(df_name):
   """
   Parameters
   ----------
   df_path : TYPE
      DESCRIPTION.

   Returns
   -------
   None.

   """
   # get correct enviroment
   load_dotenv(find_dotenv())
   # define dict which will contain all possible class lables
   dict_of_classes = {
      "Books": 0,
      "Electronics": 0,
      "Movies_and_TV": 0,
      "CDs_and_Vinyl": 0,
      "Clothing_Shoes_and_Jewelry": 0,
      "Home_and_Kitchen": 0,
      "Kindle_Store": 0,
      "Sports_and_Outdoors": 0,
      "Cell_Phones_and_Accessories": 0,
      "Health_and_Personal_Care": 0,
      "Toys_and_Games": 0,
      "Video_Games": 0,
      "Tools_and_Home_Improvement": 0,
      "Beauty": 0,
      "Apps_for_Android": 0,
      "Office_Products": 0,
      "Pet_Supplies": 0,
      "Automotive": 1,
      "Grocery_and_Gourmet_Food": 0,
      "Patio_Lawn_and_Garden": 1,
      "Baby": 0,
      "Digital_Music": 0,
      "Musical_Instruments": 0,
      "Amazon_Instant_Video": 0
    }
   # load raw csv file for given reviews at supplied path
   df = pd.read_csv(df_name, error_bad_lines=False, names=['review', 'class'])
   # drop any rows which have missing reviews, class or a class which is not in our class dict
   # print(df['review'].isnull().sum())
   nrows = df.shape[0]
   df['review'].replace('', np.nan, inplace=True)
   df.dropna(subset=['review'], inplace=True)
   df['class'].replace('', np.nan, inplace=True)
   df.dropna(subset=['class'], inplace=True)
   print('Nr. rows dropped because containing NaN:', nrows-df.shape[0])
   # print(df['review'].isnull().sum())
   nrows = df.shape[0]
   df = df[df['class'].isin(dict_of_classes)]
   print('Nr. rows dropped because class label was incorrect:', nrows-df.shape[0])
   print(df.head())
   # One hot encode class labels
   labelencoder = LabelEncoder()
   original_classes = df['class']
   df['class'] = labelencoder.fit_transform(df['class'])
   print(df.head()) 
   # split train dataset into train, validation and test sets
   train_text, temp_text, train_labels, temp_labels = train_test_split(df['review'], df['class'], 
                                                                       random_state=1, 
                                                                       test_size=0.2, 
                                                                       stratify=df['class'])
   
   
   val_text, test_text, val_labels, test_labels = train_test_split(temp_text, temp_labels, 
                                                                   random_state=1, 
                                                                   test_size=0.1, 
                                                                   stratify=temp_labels)
   tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
   max_length = 25 
   tokens_train = tokenizer.batch_encode_plus(
    train_text.tolist(),
    max_length = max_length,
    pad_to_max_length=True,
    truncation=True
    )

   # tokenize and encode sequences in the validation set
   tokens_val = tokenizer.batch_encode_plus(
    val_text.tolist(),
    max_length = max_length,
    pad_to_max_length=True,
    truncation=True
    )

   # tokenize and encode sequences in the test set
   tokens_test = tokenizer.batch_encode_plus(
    test_text.tolist(),
    max_length = max_length,
    pad_to_max_length=True,
    truncation=True
    )
   train_seq = torch.tensor(tokens_train['input_ids'])
   train_mask = torch.tensor(tokens_train['attention_mask'])
   train_y = torch.tensor(train_labels.tolist())

   val_seq = torch.tensor(tokens_val['input_ids'])
   val_mask = torch.tensor(tokens_val['attention_mask'])
   val_y = torch.tensor(val_labels.tolist())

   test_seq = torch.tensor(tokens_test['input_ids'])
   test_mask = torch.tensor(tokens_test['attention_mask'])
   test_y = torch.tensor(test_labels.tolist())
   #define a batch size
   batch_size = 32
   # wrap tensors
   train_data = TensorDataset(train_seq, train_mask, train_y)
   # sampler for sampling the data during training
   train_sampler = RandomSampler(train_data)
   # dataLoader for train set
   train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
   # wrap tensors
   val_data = TensorDataset(val_seq, val_mask, val_y)
   # sampler for sampling the data during training
   val_sampler = SequentialSampler(val_data)
   # dataLoader for validation set
   val_dataloader = DataLoader(val_data, sampler = val_sampler, batch_size=batch_size)
   # test data
   test_data = TensorDataset(test_seq, test_mask, test_y)
   test_sampler = SequentialSampler(test_data)
   test_dataloader = DataLoader(test_data, sampler = test_sampler, batch_size=batch_size)
   
   # return train_text, val_text, test_text, train_labels, val_labels, test_labels
   # return train_dataloader, val_dataloader, test_dataloader
   return test_seq, test_mask, test_y
   
       
# if __name__ == '__main__':
#     # log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
#     # logging.basicConfig(level=logging.INFO, format=log_fmt)

#     # not used in this stub but often useful for finding various files
#     #project_dir = Path(__file__).resolve().parents[2]

#     # find .env automagically by walking up directories until it's found, then
#     # load up the .env entries as environment variables
#     load_dotenv(find_dotenv())

#     clean_data()   