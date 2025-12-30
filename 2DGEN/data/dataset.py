from torch_geometric.data import Data

class CrystDataset:
    def __init__(self, path,):
        self.path=path

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]
    
if __name__ == "__main__":
    pass