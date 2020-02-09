# hooks for callbacks, taken from detectron2
import weakref
from typing import List


class HookBase:
    """
    Base class for hooks that can be registered with :class:`TrainerBase`.

    Each hook can implement 4 methods. The way they are called is demonstrated
    in the following snippet:

    .. code-block:: python

        hook.before_train()
        for iter in range(start_iter, max_iter):
            hook.before_step()
            trainer.run_step()
            hook.after_step()
        hook.after_train()

    Notes:
        1. In the hook method, users can access `self.trainer` to access more
           properties about the context (e.g., current iteration).

        2. A hook that does something in :meth:`before_step` can often be
           implemented equivalently in :meth:`after_step`.
           If the hook takes non-trivial time, it is strongly recommended to
           implement the hook in :meth:`after_step` instead of :meth:`before_step`.
           The convention is that :meth:`before_step` should only take negligible time.

           Following this convention will allow hooks that do care about the difference
           between :meth:`before_step` and :meth:`after_step` (e.g., timer) to
           function properly.

    Attributes:
        trainer: A weak reference to the trainer object. Set by the trainer when the hook is
            registered.
    """

    def before_train(self):
        """
        Called before the first iteration.
        """
        pass

    def after_train(self):
        """
        Called after the last iteration.
        """
        pass

    def before_epoch(self):
        """
        Called before the first iteration.
        """
        pass

    def after_epoch(self):
        """
        Called after the last iteration.
        """
        pass

    def before_step(self):
        """
        Called before each iteration.
        """
        pass

    def after_step(self):
        """
        Called after each iteration.
        """
        pass

    def set_trainer(self, trainer):
        self._trainer = weakref.proxy(trainer)

    @property
    def trainer(self):
        if hasattr(self, "_trainer"):
            return self._trainer


class HookMixin:
    """
    Base class for iterative trainer with hooks.

    The only assumption we made here is: the training runs in a loop.
    A subclass can implement what the loop is.
    We made no assumptions about the existence of dataloader, optimizer, model, etc.

    Attributes:
        iter(int): the current iteration.

        start_iter(int): The iteration to start with.
            By convention the minimum possible value is 0.

        max_iter(int): The iteration to end training.

        storage(EventStorage): An EventStorage that's opened during the course of training.
    """

    def __init__(self, *args, **kwargs):
        super(HookMixin, self).__init__(*args, **kwargs)
        self._hooks = []

    def register_hooks(self, hooks: List[HookBase]):
        """
        Register hooks to the trainer. The hooks are executed in the order
        they are registered.

        Args:
            hooks (list[Optional[HookBase]]): list of hooks
        """
        hooks = [h for h in hooks if h is not None]
        for h in hooks:
            assert isinstance(h, HookBase)
            h.set_trainer(self)
        self._hooks.extend(hooks)

    def _before_train(self):
        for h in self._hooks:
            h.before_train()

    def _after_train(self):
        for h in self._hooks:
            h.after_train()

    def _before_step(self):
        for h in self._hooks:
            h.before_step()

    def _after_step(self):
        for h in self._hooks:
            h.after_step()

    def _before_epoch(self):
        for h in self._hooks:
            h.before_epoch()

    def _after_epoch(self):
        for h in self._hooks:
            h.after_epoch()

    def start_training(self):
        self._before_train()
        super(HookMixin, self)._start_training()
        self._after_train()

    def train_loop(self, *args, **kwargs):
        self._before_epoch()
        super(HookMixin, self)._train_loop(*args, **kwargs)
        self._after_epoch()

    def eval_loop(self, train_loader, epoch, mode, *args, **kwargs):
        self._before_epoch()
        super(HookMixin, self)._eval_loop(*args, **kwargs)
        self._after_epoch()

    def run_step(self, *args, **kwargs):
        self._before_step()
        super(HookMixin, self)._run_step(*args, **kwargs)
        self._after_step()
