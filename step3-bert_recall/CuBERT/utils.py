import torch

def compute_kernel_bias(vecs):
    """计算kernel和bias
    vecs.shape = [num_samples, embedding_size]，
    最后的变换：y = (x + bias).matmul(kernel)
    """
    mu = vecs.mean(dim=0, keepdim=True)
    cov = torch.cov(vecs.T)
    u, s, vh = torch.linalg.svd(cov, full_matrices=False)
    W = torch.matmul(u, torch.diag(1 / torch.sqrt(s)))
    return W, -mu

def transform_and_normalize(vecs, kernel=None, bias=None):
    """应用变换，然后标准化
    """
    if kernel is not None and bias is not None:
        vecs = (vecs + bias).matmul(kernel)
    vecs = vecs / torch.sqrt(torch.sum(vecs ** 2, dim=1, keepdim=True))
    return vecs

def whitening_torch_final(x_emd, y_emd, k=16):

    W, bias = compute_kernel_bias(torch.cat([x_emd, y_emd], dim=0))
    W = W[:,:k]
    x_emd = transform_and_normalize(x_emd, W, bias)
    y_emd = transform_and_normalize(y_emd, W, bias)

    return x_emd, y_emd

def whitening_torch_final2(x_emd, y_emd, k=16):

    W, bias = compute_kernel_bias(x_emd)
    W = W[:,:k]
    x_emd = transform_and_normalize(x_emd, W, bias)

    W, bias = compute_kernel_bias(y_emd)
    W = W[:,:k]
    y_emd = transform_and_normalize(y_emd, W, bias)
    return x_emd, y_emd


if __name__ == '__main__':
    # 生成随机数据
    data = torch.rand(5, 768)

    # 计算kernel和bias
    kernel, bias = compute_kernel_bias(data)
    kernel = kernel[:, :64]

    print('kernel.shape = ', kernel.shape)
    print('bias.shape = ', bias.shape)

    # 应用变换和标准化
    data = transform_and_normalize(data, kernel, bias)
    print('data.shape = ', data.shape)