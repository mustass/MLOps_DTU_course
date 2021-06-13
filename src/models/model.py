import torch.nn as nn

class BERT_model(nn.Module):
    def __init__(self, bert, n_class=2):
      super(BERT_model, self).__init__()
      self.bert = bert
      self.n_class = n_class 
   
      # dropout layer
      self.dropout = nn.Dropout(0.1)      
      # relu activation function
      self.relu =  nn.ReLU()
      # dense layer 1
      self.fc1 = nn.Linear(768,512)     
      # dense layer 2 (Output layer)
      self.fc2 = nn.Linear(512, self.n_class)
      #softmax activation function
      self.softmax = nn.LogSoftmax(dim=1)

    #define the forward pass
    def forward(self, sent_id, mask):
      #pass the inputs to the model  
      _, cls_hs = self.bert(sent_id, attention_mask=mask, return_dict=False)     
      x = self.fc1(cls_hs)
      x = self.relu(x)
      x = self.dropout(x)
      # output layer
      x = self.fc2(x)     
      # apply softmax activation
      x = self.softmax(x)

      return x