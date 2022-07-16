# PyTorch


## Skorch


### VAE Loss

```python

class VAELoss(nn.Module):
    def __init__(self) -> None:
        super(VAELoss, self).__init__()

    def forward(self, model_output, X) -> torch.Tensor:
        """
        Comments.
        Args:
            None.
        Returns:
            None.
        """
        Xhat, mu, log_var = model_output
        KL_Divergence = - 0.5 * (1 + log_var - mu.pow(2) - log_var.exp()).mean()
        Reconstruction_Loss = F.mse_loss(Xhat, X)

        loss = Reconstruction_Loss + KL_Divergence

        return loss
```
