from torch import nn as nn
import torch
import time
import os
import pandas as pd

def dict_weights_to_vector(w):
    w = [v for v in w.values()]
    w = torch.cat([v.flatten() for v in w])
    return w


def distance_triangle(n, my_device):
    arange_matrix = torch.arange(n, device=my_device).view(-1, 1) - torch.arange(n, device=my_device).view(1, -1)
    lower_triangular = torch.tril(arange_matrix)
    return lower_triangular


def read_and_merge_csv_files(directory_path, filenames, start_date='2010-01-01', end_date='2020-12-31'):
    print(f"Reading and merging CSV files: {filenames}")
    # Initialize an empty DataFrame
    data = pd.DataFrame()
    found_files = 0
    # Iterate through the specified filenames
    for filename in filenames:
        file_path = os.path.join(directory_path, filename) + '.csv'

        if os.path.isfile(file_path) and file_path.endswith('.csv'):
            # Read the CSV file
            file_data = pd.read_csv(file_path)

            if file_data.empty:
                print(f"File {file_path} is empty")
                continue

            # Extract the 'Date' and 'Close' columns
            file_data = file_data[['Date', 'Close']]

            # Rename the 'Close' column to the file's name without the '.csv' extension
            file_data = file_data.rename(columns={'Close': filename})

            # Merge the data into the main DataFrame
            if data.empty:
                data = file_data
            else:
                data = data.merge(file_data, on='Date', how='outer')

            found_files += 1
        else:
            print(f"File {file_path} not found")

    data = data.fillna(method='bfill').reset_index(drop=True)
    data['Date'] = pd.to_datetime(data['Date'], utc=True)
    data = data.reset_index(drop=True)
    data = data.sort_values(by='Date')

    start_date = pd.Timestamp(start_date, tz='UTC')
    end_date = pd.Timestamp(end_date, tz='UTC')
    data = data.loc[(data['Date'] >= start_date) & (data['Date'] <= end_date)]
    data = data.fillna(method='bfill').reset_index(drop=True)

    return data, found_files


class TransformerConfig:

    def __init__(self,
                 my_device='cuda',
                 precision=torch.bfloat16,
                 batch_size=64,
                 block_size=32,
                 input_embed=64,
                 n_embed=64,
                 output_embed=64,
                 n_head=4,
                 n_layer=4,
                 causal=True,
                 learning_rate=1e-3
                 ):
        self.my_device = my_device
        self.causal = causal

        self.batch_size = batch_size
        self.block_size = block_size
        self.input_embed = input_embed
        self.n_embed = n_embed
        self.output_embed = output_embed
        self.hidden_size = self.n_embed * 4

        self.n_head = n_head
        self.n_layer = n_layer
        self.dropout = 0.1
        self.head_size = self.n_embed // self.n_head

        self.eval_interval = 50
        self.learning_rate = learning_rate
        self.eval_iters = 200

        self.precision = precision

        self.save_model_periodically_every_n_iterations = -1


class GeluFeedForward(nn.Module):
    def __init__(self, inp_n_embd, hidden_n_embd, out_n_embd, dropout, bias=False):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inp_n_embd, hidden_n_embd, bias=bias),
            nn.GELU(),
            nn.Linear(hidden_n_embd, out_n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TimeseriesDataloader(object):

    def __init__(self, directory_path, stocks_to_load, my_device='cuda', add_diff=True):

        df, found_files = read_and_merge_csv_files(
            directory_path,
            stocks_to_load,
            start_date='2000-01-01',
            end_date='2020-12-31'
        )

        df.drop(columns=['Date'], axis=1, inplace=True)

        prices = torch.tensor(df.values, dtype=torch.float, device=my_device)

        if add_diff:
            prices_diff = torch.diff(prices, dim=0)
            self.number_of_channels = found_files * 2
            self.data = torch.concat([prices[1:], prices_diff], dim=1)
        else:
            self.data = prices
            self.number_of_channels = found_files

        n = int(0.9 * len(self.data))  # first 90% will be trained, rest - eval
        self.train_data = self.data[:n]
        self.val_data = self.data[n:]

        self.found_files = found_files

        print("Found files: ", found_files)

    def n_channels(self):
        return self.number_of_channels

    def get_train_data(self):
        return self.train_data

    def get_val_data(self):
        return self.val_data

    def get_data(self):
        return self.data


class GenericDataloader:

    def __init__(self, config: TransformerConfig, in_data, out_data):

        self.config = config
        self.in_data = in_data.to(self.config.my_device).to(config.precision)
        self.out_data = out_data.to(self.config.my_device).to(config.precision)
        self.config = config

        n = int(0.9 * len(self.in_data))  # first 90% will be trained, rest val
        self.in_train_data = self.in_data[:n]
        self.in_val_data = self.in_data[n:]

        self.out_train_data = self.out_data[:n]
        self.out_val_data = self.out_data[n:]

    def get_batch(self, split):
        # generate a small batch of data of inputs x and targets y
        in_data = self.in_train_data if split == 'train' else self.in_val_data
        out_data = self.out_train_data if split == 'train' else self.out_val_data

        ix = torch.randint(len(in_data) - self.config.block_size, (self.config.batch_size,))

        x = torch.stack([in_data[i:i + self.config.block_size] for i in ix])
        y = torch.stack([out_data[i + 1:i + self.config.block_size + 1] for i in ix])

        x, y = x.to(self.config.my_device), y.to(self.config.my_device)

        # print(x.shape, y.shape)
        return x, y

    def get_train_batch(self):
        return self.get_batch('train')

    def get_val_batch(self):
        return self.get_batch('test')


class PositionalEmbedding(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embed)
        self.position_embedding_ff = GeluFeedForward(config.n_embed, config.n_embed, config.n_embed, config.dropout)
        self.position_embedding_ff_ln = nn.LayerNorm(config.n_embed)

    def forward(self, b, t):
        pos_embedding_arrange = torch.arange(t, device=self.config.my_device)
        pos_emb = self.position_embedding_table(pos_embedding_arrange).repeat(b, 1, 1)  # (B,T,C)
        pos_emb = self.position_embedding_ff.forward(pos_emb)
        pos_emb = self.position_embedding_ff_ln(pos_emb)
        # pos_emb = self.dropout(pos_emb)

        # pos_emb = pos_emb.unsqueeze(1).repeat(1, t, 1, 1)  # (B,T,C) -> (B,T,T,C)
        # k = pos_emb
        # q = pos_emb.transpose(1, 2)
        # pos_emb = torch.cat([k, q], dim=-1)  # (B,T,T,C)

        # return k + q
        return pos_emb


class DistancePositionalEmbedding(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embed)
        self.position_embedding_ff = GeluFeedForward(
            config.n_embed,
            config.n_embed,
            config.n_embed * 2,
            config.dropout
        )
        # self.position_embedding_ff_ln = nn.LayerNorm(config.n_embed * 2)

    def forward(self, b):
        pos_embedding_arrange = distance_triangle(self.config.block_size, self.config.my_device)
        pos_emb = self.position_embedding_table(pos_embedding_arrange)
        pos_emb = pos_emb.repeat(b, 1, 1, 1)  # (B, T, T, C)
        pos_emb = self.position_embedding_ff.forward(pos_emb)
        # pos_emb = self.position_embedding_ff_ln(pos_emb)
        return pos_emb


class AbstractRunner(object):
    def __init__(self, config: TransformerConfig, model: nn.Module, data_loader: GenericDataloader):
        self.model = model.to(config.my_device, dtype=config.precision)
        self.parameters = self.model.parameters()
        # self.model = torch.compile(model, mode="max-autotune", backend="cudagraphs") # , fullgraph=True
        self.optimizer = torch.optim.AdamW(self.parameters, lr=config.learning_rate)
        self.config = config
        self.current_iteration = 0
        self.data_loader = data_loader
        self.model_version = 1
        print(sum(p.numel() for p in self.model.parameters()) / 1e6, 'M parameters')

    def forward(self, x):
        return self.model(x)

    def learn(self, x, y):
        self.model.train()
        out, loss = self.model.forward_vs_target(x, y)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        return out, loss

    @torch.no_grad()
    def evaluate(self, get_batch, eval_iters):
        self.model.eval()
        losses = 0
        for k in range(eval_iters):
            x, y = get_batch()
            logits, loss = self.model.forward_vs_target(x, y)
            losses += loss
        return losses / eval_iters

    def train_iterate_n(self, n_iter):
        self.train_iterate(n_iter, self.data_loader.get_train_batch, self.data_loader.get_val_batch)

    def train_iterate(self, n_iter, get_train_batch, get_val_batch):
        eval_interval = self.config.eval_interval

        if self.config.eval_interval == -1:
            eval_interval = n_iter
            self.current_iteration = 1

        last_loss = 0
        t = time.time()
        for _ in range(n_iter):
            if eval_interval != -1 and self.current_iteration % eval_interval == 0:
                t_taken = time.time() - t
                train_losses = torch.sqrt(self.evaluate(get_train_batch, self.config.eval_iters))
                val_losses = torch.sqrt(self.evaluate(get_val_batch, self.config.eval_iters))
                print(
                    f"step {self.current_iteration}: rmse train loss {train_losses:.4f}, rmse val loss {val_losses:.4f}, sec/iter {t_taken / eval_interval}")

                t = time.time()

                if self.config.save_model_periodically_every_n_iterations != -1 and self.current_iteration % self.config.save_model_periodically_every_n_iterations == 0:
                    torch.save(self.model.state_dict(), f"model-{self.model_version}.pt")
                    print(f"saved model version {self.model_version}")
                    self.model_version += 1

            x, y = get_train_batch()

            logits, loss = self.learn(x, y)
            last_loss = loss.item()
            self.current_iteration += 1

        return last_loss

    def get_weights(self):
        return self.model.state_dict()

    def get_weights_as_tensor(self):
        w = self.get_weights()
        return dict_weights_to_vector(w)

    def set_weights_as_tensor(self, new_state_tensor):
        new_state_dict = self.get_weights()
        for k, v in new_state_dict.items():
            new_state_dict[k] = new_state_tensor[:v.numel()].reshape(v.shape)
            new_state_tensor = new_state_tensor[v.numel():]
        self.set_weights(new_state_dict)

    def set_weights(self, new_state_dict):
        self.model.load_state_dict(new_state_dict)

    def generate(self, *args):
        raise NotImplementedError()
