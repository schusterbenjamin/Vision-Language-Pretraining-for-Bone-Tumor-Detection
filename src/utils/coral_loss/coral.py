# NOTE: Adapted from https://github.com/DenisDsh/PyTorch-Deep-CORAL/blob/master/coral.py, credit given in ATTRIBUTION.md 
import torch


def coral(source, target):

    d = source.size(1)  # dim vector

    source_c = compute_covariance(source)
    target_c = compute_covariance(target)

    loss = torch.sum(torch.mul((source_c - target_c), (source_c - target_c)))

    loss = loss / (4 * d * d)
    return loss


def compute_covariance(input_data):
    """
    Compute Covariance matrix of the input data
    """
    n = input_data.size(0)  # batch_size

    # Check if using gpu or cpu
    if input_data.is_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    id_row = torch.ones(n).resize(1, n).to(device=device)
    sum_column = torch.mm(id_row, input_data)
    mean_column = torch.div(sum_column, n)
    term_mul_2 = torch.mm(mean_column.t(), mean_column)
    d_t_d = torch.mm(input_data.t(), input_data)
    c = torch.add(d_t_d, (-1 * term_mul_2)) * 1 / (n - 1)

    return c

if __name__ == "__main__":
    # My own little example
    source = torch.Tensor([[1.0], [1.0], [1.1], [0.9]])
    target_large_diff = torch.Tensor([[10], [10], [11]]) # note that they dont have to have the same number of samples
    target_no_diff = torch.Tensor([[1.0], [1.0], [1.1]])
    target_nan_example = torch.Tensor([[1.1]])
    loss_large = coral(source, target_large_diff)
    loss_no_diff = coral(source, target_no_diff)
    loss_nan = coral(source, target_nan_example)
    print("CORAL Loss with large difference:", loss_large.item())
    print("CORAL Loss with no difference:", loss_no_diff.item())
    print("CORAL Loss is NaN if one set only has one sample:", loss_nan.item())