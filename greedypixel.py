import random
import torch
import torch.nn.functional as F
from itertools import product

class GreedyPixel():

    def __init__(
            self,
            target,
            surrogate,
            eps: float = 4/255,
            max_query: int = 10000,
            early_stop: bool = True,
            ):
        """
        :param target: Target model.
        :param target: Surrogate model for computing gradient map.
        :param eps: Maximum perturbation.
        :param max_iter: The maximum number of iterations.
        :param early_stop: stop when label changes.
        """
        self.target=target
        self.surrogate=surrogate
        self.eps=eps
        self.max_query=max_query
        self.early_stop=early_stop
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def compute_gradient_order(self, x):
        x = x.clone().detach().to(self.device)
        if self.surrogate is None:
            # Random order: shuffle all pixel coordinates
            h, w = x.shape[-2], x.shape[-1]
            coords = [(r, c) for r in range(h) for c in range(w)]
            random.shuffle(coords)
            return coords

        # Gradient-based order
        surrogate = self.surrogate.to(self.device)
        x.requires_grad_(True)
        logits = surrogate(x)
        pred_class = logits.argmax(dim=1)
        loss = F.cross_entropy(logits, pred_class)
        surrogate.zero_grad(set_to_none=True)
        loss.backward()

        grad = x.grad.detach()
        grad_map = grad.abs().sum(dim=1).squeeze(0)  # Sum over channels
        idx = torch.argsort(grad_map.flatten(), descending=True)
        h, w = grad_map.shape
        rows = (idx // w).cpu().numpy()
        cols = (idx % w).cpu().numpy()
        return list(zip(rows, cols))

    @torch.no_grad()
    def cw_loss(self, logits, y, kappa: float = 0.0):
        """
        Carlini-Wagner (untargeted) loss: maximize logit of any class ≠ y over y.
        logits: [N, C], y: [N]
        """
        one_hot = F.one_hot(y, num_classes=logits.shape[1]).bool()
        true_logits = logits[one_hot]  # [N]
        other_logits = logits.masked_fill(one_hot, -1e9)
        max_other_logits, _ = other_logits.max(dim=1)
        # We minimize this loss → pick perturbation with smallest value
        return torch.clamp(true_logits - max_other_logits + kappa, min=0)

    def attack(self, x, y):
        x = x.clone().detach().to(self.device)
        y = y.clone().detach().to(self.device)

        pixel_order = self.compute_gradient_order(x)
        query = 0

        for it, (r, c) in enumerate(pixel_order, start=1):
            if query > self.max_query:
                break

            orig_px = x[0, :, r, c].clone()

            # Generate all 8 candidates at once
            deltas = torch.tensor(list(product([-self.eps, self.eps], repeat=3)),
                                  device=self.device)
            cand_pixels = (orig_px.unsqueeze(0) + deltas).clamp(0, 1)  # [8, 3]

            # Clone x for each candidate and apply perturbation
            x_batch = x.repeat(len(deltas), 1, 1, 1)  # [8, C, H, W]
            x_batch[:, :, r, c] = cand_pixels

            # Forward pass (batched)
            x_batch.requires_grad_(False)
            logits = self.target(x_batch)
            query += len(deltas)

            # Compute CW losses and pick the best
            losses = self.cw_loss(logits, y.repeat(len(deltas)))
            best_idx = losses.argmin()
            x[0, :, r, c] = cand_pixels[best_idx]

            # Early stop: misclassification achieved
            pred = logits[best_idx].argmax()
            if self.early_stop and pred != y.item():
                break

        return x.detach(), query