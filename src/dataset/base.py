from torch.utils.data import DataLoader, IterableDataset

from ..utils.tools import print_rank_0
class RepeatingLoader:
    def __init__(self, loader: DataLoader):
        """
        Wraps an iterator to allow for infinite iteration. This is especially useful
        for DataLoader types that we wish to automatically restart upon completion.

        Args:
            loader (DataLoader): The data loader to repeat.
        """
        self.epochs = 0
        self.loader = loader
        self.data_iter = iter(loader)
        self.global_rank = getattr(loader.dataset, "global_rank", 0)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self.data_iter)
        except StopIteration:
            self.epochs += 1
            print_rank_0(
                f"--->Start a new iteration for repeating loader for rank {self.global_rank}.",
                force_print=True
            )
            self.data_iter = iter(self.loader)
            return next(self.data_iter)

    def __len__(self):
        return len(self.loader)

    @property
    def train_token_count(self):
        """
        Only BaseDataset or its variants have this attribute.
        This value is the train token count of a single rank.
        """
        dataset = self.loader.dataset

        # If dataset is wrapped by PackingDataset or IterablePackingDataset, unwrap it.


        if isinstance(dataset, IterableDataset):
            return dataset.train_token_count
        else:
            # For non-iterable dataset, the train token count is fixed after initialization.
            print_rank_0(f"--->Return train token count after epoch: {self.epochs}", self.global_rank)
            return self.epochs * dataset.train_token_count
        return 0