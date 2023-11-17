import torch
import torch.nn as nn
    
class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, d: float = 0.5):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p = d)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        x = self.fc1(x.view(-1, self.input_dim))
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
    def predict(input_data, model):
        # make prediction
        yhat = model(input_data)
        # retrieve numpy array
        yhat = yhat.detach().numpy()
        return yhat