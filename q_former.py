import math
import torch
import torch.nn as nn


class LearnedQueries(nn.Module):
    
    def __init__(self, num_queries=32, dim_queries=768):
        super().__init__()
        self.num_queries = num_queries
        self.dim_queries = dim_queries

        self.query_tokens = nn.Parameter(torch.randn(1, num_queries, dim_queries))

    def forward(self, batch_size):
        return self.query_tokens.expand(batch_size, -1, -1)


class InputText(nn.Module):
    
    def __init__(self, dim_input, vocab_size):
        super().__init__()
        self.dim_input = dim_input
        self.vocab_size = vocab_size

        self.input_embeddings = nn.Embedding(vocab_size, dim_input)

    def forward(self, x):
        return self.input_embeddings(x) * math.sqrt(self.dim_input)


class FeedForward(nn.Module):
    
    def __init__(self, in_dim, hidden_dim, dropout):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)

        self.ff = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, in_dim)
        )

    def forward(self, x):
        return self.dropout(self.ff(x))





if __name__ == "__main__":
    learned_queries = LearnedQueries()
    input_text = InputText()

    image_features = torch.randn(4, 257, 1024)
    text_features = torch.randn()

    queries = learned_queries(4)
    print(queries.shape)

    text_embeddings = input_text(image_features)
    print(text_embeddings.shape)
