from torchmetrics import Metric
import torch
import torch.distributed as dist


def _myaccuracy_dist_sync_fn(result: torch.Tensor, group=None):
    """
    TorchMetrics state is plain tensors (not buffers), so it often stays on CPU even when
    the LightningModule is on CUDA. Default gather uses NCCL and requires CUDA tensors.
    Skip all_gather when not distributed or single-rank so CPU accumulators work.
    """
    if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() <= 1:
        return [result]
    from torchmetrics.utilities.distributed import gather_all_tensors

    return gather_all_tensors(result, group)


class MyAccuracy(Metric):
    def __init__(self, device="cpu", dist_sync_on_step=False):
        super().__init__(
            dist_sync_on_step=dist_sync_on_step,
            dist_sync_fn=_myaccuracy_dist_sync_fn,
        )

        self.add_state("correct", default=torch.tensor(0., device=device), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0., device=device), dist_reduce_fx="sum")

    def update(self, correct, total):
        if torch.is_tensor(correct):
            correct = correct.detach().to(dtype=self.correct.dtype, device=self.correct.device)
        else:
            correct = torch.tensor(
                float(correct), dtype=self.correct.dtype, device=self.correct.device
            )
        if torch.is_tensor(total):
            total = total.detach().to(dtype=self.total.dtype, device=self.total.device)
        else:
            total = torch.tensor(
                float(total), dtype=self.total.dtype, device=self.total.device
            )
        self.correct += correct
        self.total += total

    def compute(self):
        if self.total.item() == 0.:
            return self.total
        return self.correct.float() / self.total