import torch
from torch import nn
from transformers.modeling_bert import BertPreTrainedModel, BertModel, BertForQuestionAnswering

class BertLSTM(nn.Module):
    
    def __init__(self, input_size, hidden_size, bidirectional, num_layers=1, dropout=0.):
        super(BertLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            bidirectional=bidirectional,
            dropout=dropout
        )
        
    def forward(self, batch):
       # batch : B x seqlen x features
        batch = batch.transpose(1, 0)
        # now seqlen x B x features
        # output (seq_len, batch, n_directions * hidden_size)
        output, (h_n, c_n) = self.lstm(batch)
        return output.transpose(1, 0)
    
    
class BertLSTMEncoder(BertLSTM):
        
    def forward(self, batch):
       # batch : B x seqlen x features
        batch = batch.transpose(1, 0)
        # now seqlen x B x features
        # output (seq_len, batch, hidden_size)
        return self.lstm(batch)
        
        
class BertLSTMDecoder(BertLSTM):
        
    def forward(self, batch, hidden, cell):
        # batch: seqlen x B x features
        # output (seq_len, batch, n_ directions, hidden_size)
        output, (h_n, c_n) = self.lstm(batch, (hidden, cell))
        return output.transpose(1, 0)
    
class BertLSTMEncoderDecoder(nn.Module):
    
    def __init__(self, input_size, hidden_size, bidirectional, num_layers=1, dropout=0.):
        super(BertLSTMEncoderDecoder, self).__init__()
        self.bidirectional = bidirectional
        self.encoder = BertLSTMEncoder(
            input_size, 
            hidden_size, 
            bidirectional,
            num_layers, 
            dropout
        )
        
        self.decoder = BertLSTMDecoder(
            2 * hidden_size if bidirectional else hidden_size, 
            hidden_size, 
            bidirectional,
            num_layers, 
            dropout
        )

    def forward(self, batch):
        batch, (h, c) = self.encoder(batch)
        return self.decoder(batch, h, c)
   
        

class BertForQuestionAnsweringUnidirectionalLSTM(BertForQuestionAnswering):
    
    def __init__(self, config):
        super(BertForQuestionAnsweringUnidirectionalLSTM, self).__init__(config)
        self.qa_outputs = nn.Sequential(
            BertLSTM(config.hidden_size, config.hidden_size, False),
            nn.Linear(config.hidden_size, config.num_labels)
        )

        self.init_weights()

class BertForQuestionAnsweringUnidirectionalLSTM2Layer(BertForQuestionAnswering):
    
    def __init__(self, config):
        super(BertForQuestionAnsweringUnidirectionalLSTM2Layer, self).__init__(config)
        self.qa_outputs = nn.Sequential(
            BertLSTM(config.hidden_size, config.hidden_size, False, num_layers=2),
            nn.Linear(config.hidden_size, config.num_labels)
        )

        self.init_weights()
        
class BertForQuestionAnsweringUnidirectionalLSTMEncoderDecoder(BertForQuestionAnswering):
    
    def __init__(self, config):
        super(BertForQuestionAnsweringUnidirectionalLSTMEncoderDecoder, self).__init__(config)

        self.qa_outputs = nn.Sequential(
            BertLSTMEncoderDecoder(config.hidden_size, config.hidden_size, False),
            nn.Linear(config.hidden_size, config.num_labels)
        )

        self.init_weights()
        


        
class BertForQuestionAnsweringBidirectionalLSTM(BertForQuestionAnswering):
    
    def __init__(self, config):
        super(BertForQuestionAnsweringBidirectionalLSTM, self).__init__(config)

        self.qa_outputs = nn.Sequential(
            BertLSTM(config.hidden_size, config.hidden_size, True),
            nn.Linear(2 * config.hidden_size, config.num_labels)
        )

        self.init_weights()

class BertForQuestionAnsweringBidirectionalLSTM2Layer(BertForQuestionAnswering):
    
    def __init__(self, config):
        super(BertForQuestionAnsweringBidirectionalLSTM2Layer, self).__init__(config)

        self.qa_outputs = nn.Sequential(
            BertLSTM(config.hidden_size, config.hidden_size, True, num_layers=2),
            nn.Linear(2 * config.hidden_size, config.num_labels)
        )

        self.init_weights()
        
class BertForQuestionAnsweringBidirectionalLSTMEncoderDecoder(BertForQuestionAnswering):
    
    def __init__(self, config):
        super(BertForQuestionAnsweringBidirectionalLSTMEncoderDecoder, self).__init__(config)

        self.qa_outputs = nn.Sequential(
            BertLSTMEncoderDecoder(config.hidden_size, config.hidden_size, True),
            nn.Linear(2 * config.hidden_size, config.num_labels)
        )

        self.init_weights()