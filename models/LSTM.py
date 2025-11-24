import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding


class Model(nn.Module):
    """
    Vanilla LSTM for Time Series Forecasting
    Encoder-Decoder architecture for multi-horizon forecasting
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = False
        
        # Hyperparameters
        self.hidden_size = configs.d_model  # Use d_model from config for consistency
        self.num_layers = configs.e_layers
        self.dropout = configs.dropout
        
        # Embedding layer (same as other models for consistency)
        self.enc_embedding = DataEmbedding(
            configs.enc_in, 
            configs.d_model, 
            configs.embed, 
            configs.freq,
            configs.dropout
        )
        
        # LSTM Encoder
        self.encoder = nn.LSTM(
            input_size=configs.d_model,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout if self.num_layers > 1 else 0,
            batch_first=True
        )
        
        # LSTM Decoder
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.decoder = nn.LSTM(
                input_size=configs.d_model,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                dropout=self.dropout if self.num_layers > 1 else 0,
                batch_first=True
            )
            
            # Projection layer to output
            self.projection = nn.Linear(self.hidden_size, configs.c_out)
            
        # For other tasks
        if self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(self.hidden_size, configs.c_out)
            
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout_layer = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(self.hidden_size * configs.seq_len, configs.num_class)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """
        Long-term and short-term forecasting
        Args:
            x_enc: [batch_size, seq_len, enc_in]
            x_mark_enc: [batch_size, seq_len, mark_dim]
            x_dec: [batch_size, label_len+pred_len, dec_in]
            x_mark_dec: [batch_size, label_len+pred_len, mark_dim]
        Returns:
            dec_out: [batch_size, pred_len, c_out]
        """
        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B, seq_len, d_model]
        
        # LSTM Encoder
        _, (hidden, cell) = self.encoder(enc_out)  # hidden: [num_layers, B, hidden_size]
        
        # Prepare decoder input
        # Create a repeated context vector for pred_len steps
        batch_size = x_enc.shape[0]
        
        # Initialize decoder input with zeros or use embedding
        dec_inp = torch.zeros(batch_size, self.pred_len, enc_out.shape[-1]).to(x_enc.device)
        
        # LSTM Decoder with encoder's hidden state as initial state
        dec_out, _ = self.decoder(dec_inp, (hidden, cell))  # [B, pred_len, hidden_size]
        
        # Project to output dimension
        dec_out = self.projection(dec_out)  # [B, pred_len, c_out]
        
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        """
        Imputation task
        """
        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        
        # LSTM Encoder
        enc_out, _ = self.encoder(enc_out)
        
        # Projection
        dec_out = self.projection(enc_out)
        
        return dec_out

    def anomaly_detection(self, x_enc):
        """
        Anomaly detection task
        """
        # Embedding
        enc_out = self.enc_embedding(x_enc, None)
        
        # LSTM Encoder
        enc_out, _ = self.encoder(enc_out)
        
        # Projection
        dec_out = self.projection(enc_out)
        
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        """
        Classification task
        """
        # Embedding
        enc_out = self.enc_embedding(x_enc, None)
        
        # LSTM Encoder
        enc_out, _ = self.encoder(enc_out)
        
        # Output
        output = self.act(enc_out)
        output = self.dropout_layer(output)
        output = output * x_mark_enc.unsqueeze(-1)  # zero-out padding embeddings
        output = output.reshape(output.shape[0], -1)  # (batch_size, seq_length * hidden_size)
        output = self.projection(output)  # (batch_size, num_classes)
        
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, pred_len, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None
