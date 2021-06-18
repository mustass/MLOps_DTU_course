from torch import nn
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.metrics import functional as FM
import torchmetrics
from torch._C import layout
from torch.nn import functional as F
from torch.optim import Adam
import torch
import numpy as np
from transformers import AutoModel


class BERT_model(LightningModule):
    def __init__(self, full, n_class, lr):
        super().__init__()

        self.bert = AutoModel.from_pretrained('bert-base-uncased').eval()
        if not full:
            for param in self.bert.parameters():
                param.requires_grad = False


        self.n_class = n_class
        self.lr = lr

        # dropout layers
        self.dropout = nn.Dropout(0.1)
        # relu activation function
        self.relu = nn.ReLU()
        # dense layer 1
        self.fc1 = nn.Linear(768, 512)
        # dense layer 2 (Output layer)
        self.fc2 = nn.Linear(512, self.n_class)
        #softmax activation function
        self.softmax = nn.LogSoftmax(dim=1)

    #define the forward pass
    def forward(self, sent_id, mask):
        #pass the inputs to the model
        with torch.no_grad():
            _, cls_hs = self.bert(sent_id, attention_mask=mask, return_dict=False)
        
        x = self.fc1(cls_hs)
        x = self.relu(x)
        x = self.dropout(x)
        # output layer
        x = self.fc2(x)
        # apply softmax activation
        x = self.softmax(x)

        return x

    def training_step(self, batch, batch_idx):
        dat, mask, label = batch
        logits = self(dat, mask)
        loss = F.nll_loss(logits, label)
        # acc = FM.accuracy(logits, label)
        acc_metric = torchmetrics.Accuracy()
        acc = acc_metric(logits.argmax(dim=1), label)
        self.log("train_Loss",
                 loss,
                 on_step=True,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True)
        self.log("train_accuracy",
                 acc,
                 on_step=True,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True)
        
        return {"loss": loss, "outputs": logits,"accuracy": acc, "labels": label}


    def training_epoch_end(self, logits, label):
        acc_metric = torchmetrics.Accuracy()
        acc = acc_metric(logits.argmax(dim=1), label)
        self.log("train_epoch_accuracy",
                 acc,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True)

    #def training_epoch_end(self, logits, label):
    # 
    #    ps = torch.exp(logits)
    #    top_p, top_class = ps.topk(1, dim=1)
    #    equals = top_class == label.view(*top_class.shape)
    #    accuracy = torch.mean(equals.type(torch.FloatTensor))
    #
    #    accuracy = accuracy / len(label)
    #    self.log(f'Test accuracy: {accuracy.item()*100}%',on_epoch=True, logger=True)

    def validation_step(self, batch, batch_idx):
        dat, mask, label = batch
        logits = self(dat, mask)
        loss = F.nll_loss(logits, label)
        self.log("val_loss",
                 loss,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True)

    def test_step(self, batch, batch_idx):
        dat, mask, label = batch
        logits = self(dat, mask)
        loss = F.nll_loss(logits, label)
        self.log("test_loss",
                 loss,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)