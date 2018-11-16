import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np

# - - - Image Encoder - - -
class EncoderCNN(nn.Module):
    def __init__(self, embed_size = 1024):
        super(EncoderCNN, self).__init__()
        
        # get the pretrained vgg19 model
        vgg = models.vgg19_bn(pretrained=True)
        
        # freeze the gradients to avoid training
        for i, param in enumerate(vgg.parameters()):
            if i < 35:
                param.requires_grad_(False)
        
        # transfer learning procedure
        # take everything before the 34th layer of the vgg
        modules = list(vgg.children())[0][:49]
        self.vgg = nn.Sequential(*modules)
        self.embed = nn.Linear(in_features=196, out_features = embed_size)
    
    def forward(self, images):
        features = self.vgg(images)
        
        # flat the feature vector
        features = features.view(features.size(0), 512, -1)
        
        # embed the feature vector
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        
        # define the properties
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        
        # lstm cells
        self.lstm_cell_1 = nn.LSTMCell(input_size=embed_size, hidden_size=hidden_size)
        self.lstm_cell_2 = nn.LSTMCell(input_size=hidden_size, hidden_size=hidden_size)
        
        # output layer of the lstm cell
        self.concat = nn.Linear(in_features=hidden_size*2, out_features=hidden_size)
        self.fc_out = nn.Linear(in_features=hidden_size, out_features=vocab_size)
    
        # embedding layer for work embeddings
        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size)
    
        # log softmax activation for neglog loss
        self.logsoft = nn.LogSoftmax(dim=1)
    
        # softmax activation
        self.softmax = nn.Softmax(dim=1)
    
        # dropout layer
        self.drop = nn.Dropout(p=0.5)
    
    def forward(self, features, captions):
        
        # batch size
        batch_size = features.size(0)
        
        # init the hidden state to zeros
        hidden_state_1 = torch.zeros((batch_size, self.hidden_size)).cuda()
        cell_state_1 = torch.zeros((batch_size, self.hidden_size)).cuda()
        hidden_state_2 = torch.zeros((batch_size, self.hidden_size)).cuda()
        cell_state_2 = torch.zeros((batch_size, self.hidden_size)).cuda()
    
        # define the output tensor placeholder
        outputs = torch.empty((batch_size, captions.size(1), self.vocab_size)).cuda()

        # embed the captions
        captions_embed = self.embed(captions)
        
        # pass the caption word by word
        for t in range(captions.size(1)):

            # for the first time step the input is the <start> token
            if t == 0:
                
                hidden_state_1, cell_state_1 = self.lstm_cell_1(captions_embed[:, t, :], (hidden_state_1, cell_state_1))
                hidden_state_2, cell_state_2 = self.lstm_cell_2(hidden_state_1, (hidden_state_2, cell_state_2))
                  

            else:
                
                hidden_state_1, cell_state_1 = self.lstm_cell_1(captions_embed[:, t, :], (hidden_state_1, cell_state_1))
                hidden_state_2, cell_state_2 = self.lstm_cell_2(hidden_state_1, (hidden_state_2, cell_state_2))
            
            # apply dropout to the hidden state
            hidden_state_2 = self.drop(hidden_state_2)
            
            # - - - define the attention mechanics - - -
            # Dot product of the feature vector and the hidden state vector
            proxy_1 = self.softmax(torch.bmm(features, hidden_state_2.unsqueeze(2)))
        
            # element-wise multiplication of the proxy vector and the feature vector
            proxy_2 = features * proxy_1
            
            # context vector is the summation across the filter dimension
            context = torch.sum(proxy_2, dim=1)
            
            # concatenate the context vector and the hidden vector
            contextcat = torch.cat((context, hidden_state_2), dim=1)
            
            # embed the concatenation into the vocabulary space
            concat = self.drop(torch.tanh(self.concat(contextcat)))
            
            # output of the attention mechanism
            out = self.fc_out(concat)
            
            # get the top word
#             top_word = torch.argmax(out, dim=1)
            
            # construct the outputs vector
            outputs[:, t, :] = self.logsoft(out)

        return outputs

    # Todo: implement the sample method
    def sample(self, input, features, max_len=30):
        
        # init the hidden and cell states
        hidden_state_1 = torch.zeros((1, self.hidden_size)).cuda()
        cell_state_1 = torch.zeros((1, self.hidden_size)).cuda()
        hidden_state_2 = torch.zeros((1, self.hidden_size)).cuda()
        cell_state_2 = torch.zeros((1, self.hidden_size)).cuda()
        
        # embed the inputs
        embed = self.embed(input)
        
        # define the output tensor placeholder
        outputs = torch.empty((1, max_len, 1)).cuda()
        
        # main sample loop
        for t in range(max_len):
            
            # run through the LSTM
            hidden_state_1, cell_state_1 = self.lstm_cell_1(embed, (hidden_state_1, cell_state_1))
            hidden_state_2, cell_state_2 = self.lstm_cell_2(hidden_state_1, (hidden_state_2, cell_state_2))
        
            # - - - define the attention mechanics - - -
            # Dot product of the feature vector and the hidden state vector
            proxy_1 = self.softmax(torch.bmm(features, hidden_state_2.unsqueeze(2)))
        
            # element-wise multiplication of the proxy vector and the feature vector
            proxy_2 = features * proxy_1
            
            # context vector is the summation across the filter dimension
            context = torch.sum(proxy_2, dim=1)
            
            # concatenate the context vector and the hidden vector
            contextcat = torch.cat((context, hidden_state_2), dim=1)
            
            # embed the concatenation into the vocabulary space
            concat = torch.tanh(self.concat(contextcat))
            
            # output of the attention mechanism
            out = self.fc_out(concat)
            
            # embed the top word
            # batch dimension is 1
            embed = self.embed(torch.argmax(out, dim=1))
            
            # construct the outputs vector
            outputs[:, t, :] = torch.argmax(out, dim=1)
        
#             if torch.argmax(out, dim=1) == 1:
#                 return outputs
            
        return outputs