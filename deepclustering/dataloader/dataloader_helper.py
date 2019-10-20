"""
Based on https://github.com/justheuristic/prefetch_generator
"""

import queue as Queue
import threading
import warnings
from copy import deepcopy as dcopy
from typing import Union, List, Any

from torch.utils.data import DataLoader


class BackgroundGenerator(threading.Thread):
    def __init__(self, generator, max_prefetch=1):
        """

        This function transforms generator into a background-thead generator.
        :param generator: generator or genexp or any
        It can be used with any minibatch generator.

        It is quite lightweight, but not entirely weightless.
        Using global variables inside generator is not recommended (may rise GIL and zero-out the benefit of having a background thread.)
        The ideal use case is when everything it requires is store inside it and everything it outputs is passed through queue.

        There's no restriction on doing weird stuff, reading/writing files, retrieving URLs [or whatever] wlilst iterating.

        :param max_prefetch: defines, how many iterations (at most) can background generator keep stored at any moment of time.
        Whenever there's already max_prefetch batches stored in queue, the background process will halt until one of these batches is dequeued.

        !Default max_prefetch=1 is okay unless you deal with some weird file IO in your generator!

        Setting max_prefetch to -1 lets it store as many batches as it can, which will work slightly (if any) faster, but will require storing
        all batches in memory. If you use infinite generator with max_prefetch=-1, it will exceed the RAM size unless dequeued quickly enough.
        """
        threading.Thread.__init__(self)
        self.queue = Queue.Queue(max_prefetch)
        self.generator = generator
        self.daemon = True
        self.start()

    def run(self):
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def next(self):
        next_item = self.queue.get()
        # print(self.queue.qsize())
        if next_item is None:
            raise StopIteration
        return next_item

    # Python 3 compatibility
    def __next__(self):
        return self.next()

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.generator)


# decorator
class background:
    def __init__(self, max_prefetch=1):
        self.max_prefetch = max_prefetch

    def __call__(self, gen):
        def bg_generator(*args, **kwargs):
            return BackgroundGenerator(
                gen(*args, **kwargs), max_prefetch=self.max_prefetch
            )

        return bg_generator


# DataLoader helper function
class DataIter(object):
    def __init__(self, dataloader: Union[DataLoader, List[Any]]) -> None:
        super().__init__()
        # there should have no deep copy. since we only use the iterator of the dataloader, each iterator, it generates new objects
        self.dataloader = dataloader
        self.iter_dataloader = iter(dataloader)
        self.cache = None

    def __iter__(self):
        return self

    def __next__(self):
        try:
            self.cache = self.iter_dataloader.__next__()
            return self.cache
        except StopIteration:
            self.iter_dataloader = iter(self.dataloader)
            self.cache = self.iter_dataloader.__next__()
            return self.cache

    def __cache__(self):
        if self.cache is not None:
            return self.cache
        else:
            warnings.warn("No cache found, iterator forward")
            return self.__next__()
