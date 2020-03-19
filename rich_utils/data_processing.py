import logging

from torch.utils.data import Dataset, DataLoader

from rich_utils.data_preprocessing import dll_columns, raw_feature_columns, weight_col

logger = logging.getLogger('main.data_processing')


class ParticleSet(Dataset):
    def __init__(self, data):
        """
        arg: data - pd.DataFrame
        """
        self.data = data
        logger.info(f"ParticleSet initialized, data.shape {data.shape}, \n" +
                    f"raw_feature_columns - {raw_feature_columns}, \n " +
                    f"weight_col - {weight_col}, \n " +
                    f"dll_columns - {dll_columns}")
        self.raw_feature = data.loc[:, raw_feature_columns].values
        self.weight = data.loc[:, weight_col].values
        self.dll = data.loc[:, dll_columns].values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, ind):
        return (self.raw_feature[ind],  # X
                self.weight[ind],  # Weight
                self.dll[ind],  # DLL
                )


def get_loader(data, *dloader_args, **dloader_kwargs):
    logger.info(f"DataLoader initialized, dloader_args: {dloader_args} \n" +
                f"dloader_kwargs: {dloader_kwargs}")
    return DataLoader(ParticleSet(data), *dloader_args, **dloader_kwargs)
