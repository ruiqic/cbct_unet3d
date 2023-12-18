import torch
import torch.nn as nn
from monai.transforms import Resize
from monai.losses import GeneralizedDiceLoss

class DeepSupervisedGeneralizedDiceCELoss(nn.Module):
    def __init__(self, weight=None):
        """
        Custom loss function for deep supervision with different resolutions and weights.
        
        Args:
            weight (torch.Tensor): weight argument for nn.CrossEntropyLoss
        """
        super(GeneralizedDiceCELoss, self).__init__()
        # normalized array of [1/16, 1/8, 1/4, 1/2, 1]
        
        self.res_weights = [0.03225806, 0.06451613, 0.12903226, 0.25806452, 0.51612903]


        self.dice_loss_fn = GeneralizedDiceLoss(include_background=True, softmax=True, reduction="mean",
                                               w_type="square", to_onehot_y=True)
        self.ce_loss_fn = nn.CrossEntropyLoss(weight=weight)

    def forward(self, outputs, targets):
        """
        Compute the loss.
        
        Args:
            outputs (list of torch.Tensor): List of output predictions at different resolutions.
                Each tensor in the list has shape (batch_size, num_classes, H, W, D).
            targets (torch.Tensor): Ground truth target values.
                Shape: (batch_size, H, W, D)
        
        Returns:
            torch.Tensor (or MONAI metatensor): Loss value.
        """
        num_resolutions = len(outputs)
        assert(num_resolutions == len(self.res_weights))
        total_loss = 0
        
        for i in range(num_resolutions):
            output = outputs[i]
            weight = self.res_weights[i]
            resizer = Resize(spatial_size=output.size()[2:], mode="nearest-exact")
            scaled_targets = resizer(targets).long()
            dice_loss = self.dice_loss_fn(output, scaled_targets.unsqueeze(1))
            ce_loss = self.ce_loss_fn(output, scaled_targets)
            loss = (dice_loss + ce_loss) / 2
            total_loss += weight * loss
        
        return total_loss

    
class GeneralizedDiceCELoss(nn.Module):
    def __init__(self, weight=None):
        """
        Custom loss function combining generalized Dice loss and CE.
        
        Args:
            weight (torch.Tensor): weight argument for nn.CrossEntropyLoss
        """
        super(GeneralizedDiceCELoss, self).__init__()

        self.dice_loss_fn = GeneralizedDiceLoss(include_background=True, softmax=True, reduction="mean",
                                               w_type="square", to_onehot_y=True)
        self.ce_loss_fn = nn.CrossEntropyLoss(weight=weight)

    def forward(self, outputs, targets):
        """
        Compute the loss.
        
        Args:
            outputs (list of torch.Tensor): List of output predictions at different resolutions.
                Each tensor in the list has shape (batch_size, num_classes, H, W, D).
            targets (torch.Tensor): Ground truth target values.
                Shape: (batch_size, H, W, D)
        
        Returns:
            torch.Tensor (or MONAI metatensor): Loss value.
        """
        
        dice_loss = self.dice_loss_fn(outputs, targets.unsqueeze(1))
        ce_loss = self.ce_loss_fn(outputs, targets)
        loss = (dice_loss + ce_loss) / 2
        
        return loss
