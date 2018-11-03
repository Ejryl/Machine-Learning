import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np

class EncoderCNN(nn.Module):
    def __init__(self, embed_size = 1024):
        super(EncoderCNN, self).__init__()
        
        # get the pretrained vgg19 model
        vgg = models.vgg19(pretrained=True)
        
        # freeze the gradients to avoid training
        for param in vgg.parameters():
            param.requires_grad_(False)
        
        # transfer learning procedure
        modules = list(vgg.children())[0][:34]
        self.vgg = nn.Sequential(*modules)
        self.embed = nn.Linear(in_features=196, out_features = embed_size)
    
    def forward(self, images):
        features = self.vgg(images)
        features = features.view(features.size(0), 512, -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        
        # define the properties
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        
        # define the embedding layer for the inputs
        self.embed = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embed_size)
        
        # define the main lstm cells
        self.lstm_cell_1 = nn.LSTMCell(input_size=self.embed_size, hidden_size=self.hidden_size)
        self.lstm_cell_2 = nn.LSTMCell(input_size=self.hidden_size, hidden_size=self.hidden_size)
        
        self.context_fc = nn.Linear(in_features=2048, out_features=2048)  # Lz
        self.hidden_fc = nn.Linear(in_features=2048, out_features=2048)  # Lh
        
        # define the fully connected layer for output
        self.fc_out = nn.Linear(in_features=2048, out_features=self.vocab_size)
        
        # dropout layer
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, features, captions):
        
        """
            Features are the (batch_size, 512, 1024) dimensions
            Captions are the (batch_size, caption_length) dimensions
        """
        
        # init the hidden state to zeros
#         hidden_state = torch.uniform((captions.size(0), self.hidden_size)).cuda()
        
        hidden_state_1 = torch.mean(features, dim=1)
        hidden_state_2 = torch.empty((captions.size(0), self.hidden_size)).uniform_().cuda()
    
#         # init the cell state to zeros
#         cell_state = torch.uniform((captions.size(0), self.hidden_size)).cuda()
        cell_state_1 = torch.mean(features, dim=1)
        cell_state_2 = torch.empty((captions.size(0), self.hidden_size)).uniform_().cuda()
    
        # define the output tensor placeholder
        outputs = torch.empty((captions.size(0), captions.size(1), self.vocab_size)).cuda()
        
        # embed the captions
        captions = self.embed(captions)

        # pass the caption word by word
        for t in range(captions.size(1)):

            # for the first time step the input is the <start> token
            if t == 0:
                           
                hidden_state_1, cell_state_1 = self.lstm_cell_1(captions[:, t, :], (hidden_state_1, cell_state_1))
                hidden_state_2, cell_state_2 = self.lstm_cell_2(hidden_state_1, (hidden_state_2, cell_state_2))
                
                S = hidden_state_2
                
            else:
                
                # embed the previous output
                inputs = self.embed(torch.argmax(out, dim=1))
                
                # for the second+ step we pass the embedded context vector
                hidden_state_1, cell_state_1 = self.lstm_cell_1(inputs, (hidden_state_1, cell_state_1))
                hidden_state_2, cell_state_2 = self.lstm_cell_2(hidden_state_1, (hidden_state_2, cell_state_2))
                
                
                # - - define the attention mechanics - -
                # raw alpha scores as dot product attention mechanics
                alpha_raw = torch.bmm(hidden_state_2.unsqueeze(1), features.permute(0, 2, 1))

                # alpha scores
                alpha = F.softmax(alpha_raw, dim=2)

                # pre-context: multiply each image representation by a score
                pre_context = torch.mul(features.permute(0, 2, 1), alpha)

                # context vector: sum along the image rep dimension to identify the dominant 
                # image representation
                context = torch.sum(pre_context, dim=2)

                LzZt = self.context_fc(context)
                LhHt = self.hidden_fc(hidden_state_2)
                E = self.embed(torch.argmax(out, dim=1))
                S = LzZt + LhHt + E
            
            # concatenate the context vector and the hidden state
            #  - - - try concatinating with the cell state? - - - 
#             concat = torch.cat((context, hidden_state_2), dim=1)
                
            # pass the output of the lstm cell through the fully connected layer
            out = self.fc_out(S)
            
            # construct the outputs vector
            outputs[:, t, :] = out

        return outputs

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        
        # pass input into an lstm cell -> generate <start> token
        # keep passing the outputs of the lstm to the next cell until we encounter the <end> token
        
        # empty list to store the output sequence
        output = np.empty(0, dtype=int)
        
        # if the initial states are not specified
        # init them to zeros
        if not states:
            initial_states = (torch.zeros((1, self.hidden_size)).cuda(),
                              torch.zeros((1, self.hidden_size)).cuda())
        
        for t in range(max_len):

            # first imput is the feature vector
            if t == 0:
                
                # pass the input and states to the lstm cell
                hidden_state, cell_state = self.lstm_cell(inputs, initial_states)

            else:
                
                # pass the inputs through a fully connected embedding layer
                inputs = self.embed(inputs)
                
                # pass the input and states to the lstm cell
                hidden_state, cell_state = self.lstm_cell(inputs, (hidden_state, cell_state))
        
            # pass the hidden state through a fully connected layer
            # get the output vector of the vocabulary size
            fc_output = F.softmax(self.fc_out(hidden_state), dim=1)

            # get top 5 words from the vocabulary
            values, top_k_indices = torch.topk(fc_output, 10)

            # squeeze the batch dimension on indices
            top_k_indices = top_k_indices.squeeze().cpu().numpy()

            # squeeze the batch dimension on values
            values = values.detach().squeeze().cpu().numpy()

            # sample the index of the top word with the corresponding probability
            top_word = np.random.choice(top_k_indices, p=values/values.sum())
            
            # append the index of the top chosen word to the output sequence
            output = np.concatenate((output, np.array([top_word])))
            
            # if the sampled word is <end> -> return 
            if top_word == 1:
                return [np.asscalar(np.array(num)) for num in output]

            top_word = np.expand_dims(np.array([top_word]), 0)
            
            # if not <end> -> prepare the word for the input to the lstm cell
#             top_word = to_categorical(top_word, n_labels=self.vocab_size)

            # set the input to be a one-hot torch tensor with batch dimension of 1
            inputs = torch.from_numpy(top_word).squeeze(0).cuda()

        # if never encounter the <end> token -> return
        return [np.asscalar(np.array(num)) for num in output]
        
        
        