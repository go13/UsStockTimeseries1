import pandas as pd
import os
import torch
from transformer_common import TransformerConfig



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
            print(f"Merged {file_path}, found files={found_files}")
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
