import torch

def mutual_info_regression(x, y, n_neighbors=3):
    """
    Using PyTorch to calculate the mutual information between continuous target variable y and feature x without relying on sklearn at all.
    """
    def knn_distances(x, k):
        """
        Calculate the K nearest neighbor distance between given data points X.
        """
        n = x.size(0)
        # Calculate the distance between points
        xx = x.pow(2).sum(1, keepdim=True).expand(n, n)
        dist = xx + xx.t() - 2 * x.mm(x.t())
        # Ensure that the diagonal is infinite, and avoid becoming the nearest neighbor.
        dist = dist + torch.eye(n).to(x.device) * 1e10
        # Take the first k minimum values of each row, that is, the distance between the k nearest neighbors.
        distances, _ = dist.topk(k=k, largest=False, dim=1)
        return distances[:, -1]  # Returns the distance of the k nearest neighbor.

    # Make sure that x and y are 2D and 1D tensors.
    if len(x.shape) == 1:
        x = x.reshape(-1, 1)
    if len(y.shape) > 1:
        y = y.flatten()

    # Combine x and y to calculate their joint distribution.
    xy = torch.cat((x, y.reshape(-1, 1)), dim=1)

    rho = 0.4  # the constant for distance conversion
    dx = knn_distances(x, n_neighbors) * rho
    dy = knn_distances(y.reshape(-1, 1), n_neighbors) * rho
    dxy = knn_distances(xy, n_neighbors) * rho

    # Computing mutual information
    mi = torch.digamma(torch.tensor(n_neighbors)) + torch.digamma(torch.tensor(len(x))) - (torch.mean(torch.digamma(dx+1)) + torch.mean(torch.digamma(dy+1)) - torch.mean(torch.digamma(dxy+1)))

    return mi.item()  # Returns a Python scalar