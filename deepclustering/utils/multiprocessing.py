from multiprocessing import Queue, JoinableQueue

import torch
from PIL import Image
from torch import multiprocessing
from torchvision.transforms import Compose, Resize, ToTensor, ColorJitter

torch.set_num_threads(1)
T = Compose([Resize((224, 224)), ColorJitter(brightness=[0.8, 1.6]), ToTensor()])


def read_img(
    path_queue: multiprocessing.JoinableQueue, data_queue: multiprocessing.SimpleQueue
):
    torch.set_num_threads(1)
    while True:
        img_path = path_queue.get()
        img = Image.open(img_path)
        data_queue.put(T(img))
        path_queue.task_done()


def read_img2(img_path):
    img = Image.open(img_path)
    return T(img)


class multiprocessing_mapping(object):
    def __init__(self, num_workers=4, transform=read_img) -> None:
        super().__init__()
        self.num_workers = num_workers
        self.transform = transform
        self.data_queue = Queue()
        self.path_queue = JoinableQueue()
        self.path_queue.cancel_join_thread()
        self.workers = [
            multiprocessing.Process(
                target=self.transform, args=(self.path_queue, self.data_queue)
            )
            for _ in range(self.num_workers)
        ]

        for w in self.workers:
            w.daemon = True  # ensure that the worker exits on process exit
            w.start()

    def __call__(self, img_path_list):
        for i in img_path_list:
            self.path_queue.put(i)
        self.path_queue.join()
        return [self.data_queue.get() for _ in range(len(img_path_list))]
