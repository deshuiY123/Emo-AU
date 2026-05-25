import math


class LinearWarmupCosineLRScheduler:
    def __init__(
        self,
        optimizer,
        max_epoch,
        iters_per_epoch,
        min_lr,
        init_lr,
        warmup_steps=0,
        warmup_start_lr=-1,
        **kwargs,
    ):
        self.optimizer = optimizer
        self.max_epoch = max_epoch
        self.iters_per_epoch = iters_per_epoch
        self.min_lr = min_lr
        self.init_lr = init_lr
        self.warmup_steps = warmup_steps
        self.warmup_start_lr = warmup_start_lr if warmup_start_lr >= 0 else init_lr

        # 给每个 param group 保存自己的 lr 配置。
        # 如果 group 里没有显式指定，就回退到全局配置。
        for group in self.optimizer.param_groups:
            group.setdefault("init_lr", group.get("lr", self.init_lr))
            group.setdefault("min_lr", self.min_lr)
            group.setdefault("warmup_start_lr", self.warmup_start_lr)

    def step(self, cur_epoch, cur_step):
        """Update LR by global step instead of resetting warmup every epoch."""
        total_cur_step = cur_epoch * self.iters_per_epoch + cur_step
        total_max_step = self.max_epoch * self.iters_per_epoch

        for group in self.optimizer.param_groups:
            init_lr = group.get("init_lr", self.init_lr)
            min_lr = group.get("min_lr", self.min_lr)
            warmup_start_lr = group.get("warmup_start_lr", self.warmup_start_lr)

            if self.warmup_steps > 0 and total_cur_step < self.warmup_steps:
                lr = linear_warmup_lr(
                    step=total_cur_step,
                    max_step=self.warmup_steps,
                    start_lr=warmup_start_lr,
                    target_lr=init_lr,
                )
            else:
                lr = cosine_lr(
                    step=min(total_cur_step, total_max_step),
                    max_step=max(total_max_step, 1),
                    init_lr=init_lr,
                    min_lr=min_lr,
                )

            group["lr"] = lr

    def get_last_lr(self):
        return [param_group["lr"] for param_group in self.optimizer.param_groups]


def cosine_lr(step, max_step, init_lr, min_lr):
    return (init_lr - min_lr) * 0.5 * (
        1.0 + math.cos(math.pi * step / max(max_step, 1))
    ) + min_lr


def linear_warmup_lr(step, max_step, start_lr, target_lr):
    return min(
        target_lr,
        start_lr + (target_lr - start_lr) * step / max(max_step, 1),
    )


# 保留旧函数名，避免其他地方 import 出错。
def cosine_lr_schedule(optimizer, epoch, max_epoch, init_lr, min_lr):
    lr = cosine_lr(epoch, max_epoch, init_lr, min_lr)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def warmup_lr_schedule(optimizer, step, max_step, init_lr, max_lr):
    lr = linear_warmup_lr(step, max_step, init_lr, max_lr)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
