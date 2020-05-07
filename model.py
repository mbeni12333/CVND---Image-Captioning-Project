import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

import numpy as np

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, drop_prob=0.2):
        
        super(DecoderRNN, self).__init__()
        
        
        self.drop_prob = drop_prob
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        
        
        # layer to embed the tokenized words into vectors
        self.embedding_layer = nn.Embedding(vocab_size, embed_size)
        
        # lstm model
        self.lstm = nn.LSTM(input_size=embed_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            dropout=drop_prob,
                            batch_first=True)
        # dropout layer to avoid overfitting
        self.dropout = nn.Dropout(drop_prob)
        # Linear layer 
        self.fc = nn.Linear(hidden_size, vocab_size)
        # initialize weights
        self.init_weights()
     
    def init_weights(self):
        
        # initilize embeding layer
        nn.init.xavier_uniform_(self.embedding_layer.weight)
        # initialize lstm weights
        
#         for param in self.lstm.parameters():
#             nn.init.xavier_uniform_(param)
        
        # initialize weights of the fc layer
        nn.init.xavier_uniform_(self.fc.weight)
        
        
    def forward(self, features, captions):
        """
        Forward pass
        """
        # discard the <end> token
        captions = captions[:, :-1]
        # calculate the embedding
        captions = self.embedding_layer(captions)
        # cancatenate the features with the embeded captions
        inputs = torch.cat((features.unsqueeze(1), captions), dim=1)
        # lstm_outputs 
        outputs, _ = self.lstm(inputs) #shape (batch_size, caption_len, hidden_size)
        
        # convert lstm outputs to vector of distibution
        outputs = self.fc(outputs) #shape (batch_size, caption_len, vocab_size)
        
        return outputs

    ## Greedy search 

    def sample(self, inputs, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        

        outputs = []
        outputs_random = []
        current_len = 0
        hidden = None

        with torch.no_grad():
            
            while (current_len != max_len):

                # feed the inputs to the lstm
                output, hidden = self.lstm(inputs, hidden)

                #convert to vector ov vocab size
                output = self.fc(output.squeeze(1))
                # choose index greedily
                _, max_index = torch.max(output, 1)
                outputs.append(max_index.cpu().numpy()[0].item())
                
                #print(max_index.dtype)
                #distribution = F.softmax(output, 1).cpu().numpy().squeeze()
                #random_idx = np.random.choice(np.arange(self.vocab_size), p=distribution)
                #random_idx = torch.tensor(random_idx, dtype=torch.int64).unsqueeze(0).to("cuda")
                
                # if we encounter <end> we stop
                if (max_index == 1):
                    break

                # pass the predicted word to embedding_layer for the next iteration
                inputs = self.embedding_layer(max_index) 
                # add the batch dimension
                inputs = inputs.unsqueeze(1)

                current_len += 1

            return outputs
    
    
