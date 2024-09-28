import torch

def cosine_similarity_matrix(x, y):
    x_norm = x / x.norm(dim=1, keepdim=True)
    y_norm = y / y.norm(dim=1, keepdim=True)
    return torch.mm(x_norm, y_norm.t())

if __name__ == '__main__':
    x = torch.rand(5, 128)  
    y = torch.rand(6, 128)
    similarity_matrix = cosine_similarity_matrix(x, y)
    print(similarity_matrix)