import torch

"""
Adapted from https://github.com/neuraloperator/neuraloperator/blob/main/neuralop/losses/meta_losses.py to weighted Fieldwise sum.
"""


class FieldwiseAggregatorLoss(object):
    """
    AggregatorLoss takes a dict of losses, keyed to correspond
        to different properties or fields of a model's output.
        It then returns an aggregate of all losses weighted by
        an optional weight dict.

    params:
        losses: dict[Loss]
            a dictionary of loss functions, each of which
            takes in some truth_field and pred_field
        mappings: dict[tuple(Slice)]
            a dictionary of mapping indices corresponding to
            the output fields above. keyed 'field': indices,
            so that pred[indices] contains output for specified field
        logging: bool
            whether to track error for each output field of the model separately

    """

    def __init__(self, losses: dict, mappings: dict, logging=False, weights=None):
        # AggregatorLoss should only be instantiated
        # with more than one loss.
        assert (
            mappings.keys() == losses.keys()
        ), "Mappings \
               and losses must use the same keying"
        if weights is not None:
            assert len(weights) == len(
                mappings
            ), "Different Number of Weights and Fields"
        else:
            # create a list of 1/ len
            weights = [1.0 / len(mappings)] * len(mappings)
        self.losses = losses
        self.mappings = mappings
        self.logging = logging
        self.weights = weights

    def __call__(self, pred: torch.Tensor, truth: torch.Tensor, **kwargs):
        """
        Calculate aggregate loss across model inputs and outputs.

        parameters
        ----------
        pred: tensor
            contains predictions output by a model, indexed for various output fields
        y: tensor
            contains ground truth. Indexed the same way as pred.
        **kwargs: dict
            bonus args to pass to each fieldwise loss
        """

        loss = 0.0
        if self.logging:
            loss_record = {}
        ind = 0
        # sum losses over output fields
        for field, indices in self.mappings.items():
            pred_field = pred[indices]
            truth_field = truth[indices]
            field_loss = self.losses[field](pred_field, truth_field, **kwargs)
            loss += self.weights[ind] * field_loss
            if self.logging:
                loss_record[field] = field_loss
            ind += 1
        loss = loss

        if self.logging:
            return loss, loss_record
        else:
            return loss


class WeightedSumLoss(object):
    """
    Computes an average or weighted sum of given losses.
    """

    def __init__(self, losses, weights=None):
        super().__init__()
        if weights is None:
            weights = [1.0 / len(losses)] * len(losses)
        if not len(weights) == len(losses):
            raise ValueError("Each loss must have a weight.")
        self.losses = list(zip(losses, weights))

    def __call__(self, *args, **kwargs):
        weighted_loss = 0.0
        for loss, weight in self.losses:
            weighted_loss += weight * loss(*args, **kwargs)
        return weighted_loss

    def __str__(self):
        description = "Combined loss: "
        for loss, weight in self.losses:
            description += f"{loss} (weight: {weight}) "
        return description
