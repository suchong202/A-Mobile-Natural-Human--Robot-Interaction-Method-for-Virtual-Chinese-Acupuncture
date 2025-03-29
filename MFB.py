import numpy as np
import torch
import torch.nn as nn

class MFB(nn.Module):
    def __init__(self, in_dim1, in_dim2, out_dim, factor_dim):
        super(MFB, self).__init__()
        self.in_dim1 = in_dim1
        self.in_dim2 = in_dim2
        self.out_dim = out_dim
        self.factor_dim = factor_dim

        # Linear transformations for the two input modalities
        self.linear1 = nn.Linear(in_dim1, factor_dim)
        self.linear2 = nn.Linear(in_dim2, factor_dim)

        # Linear transformation for the output
        self.linear_out = nn.Linear(factor_dim * factor_dim, out_dim)

        # Dropout layer (optional)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x1, x2):
        # x1 and x2 are the input features from the two modalities
        # Shape of x1: (batch_size, in_dim1)
        # Shape of x2: (batch_size, in_dim2)

        # Apply linear transformations
        x1_transformed = self.linear1(x1)  # Shape: (batch_size, factor_dim)
        x2_transformed = self.linear2(x2)  # Shape: (batch_size, factor_dim)

        # Compute the outer product (bilinear pooling)
        outer_product = torch.bmm(x1_transformed.unsqueeze(2), x2_transformed.unsqueeze(1))  # Shape: (batch_size, factor_dim, factor_dim)

        # Flatten the outer product to a 2D tensor
        flattened = outer_product.view(outer_product.size(0), -1)  # Shape: (batch_size, factor_dim * factor_dim)

        # Apply dropout (optional)
        flattened = self.dropout(flattened)

        # Apply the final linear transformation
        output = self.linear_out(flattened)  # Shape: (batch_size, out_dim)

        return output


def Mfb(f1,f2):

    x1 = torch.tensor(f1).float()
    x2 = torch.tensor(f2).float()

    in_dim=len(x1[0])

    out_dim = 9  # Desired output dimension (e.g., for classification)
    factor_dim = 32  # Factor dimension for MFB

    # Create an MFB instance
    mfb = MFB(in_dim, in_dim, out_dim, factor_dim)

    # Forward pass through MFB
    output = mfb(x1, x2)
    print("Output shape:", output.tolist())


    return output.tolist()

# Example usage
if __name__ == "__main__":
    f1 = [[1.1, 3.1, 4.1, 6.1, 2.1, 4.1, 7.1, 8.1, 4.1]]
    f2 = [[1.1, 3.1, 4.1, 6, 2.1, 4.1, 7.1, 8.1, 4.1]]





    #print(x1,x2)
    Mfb(f1,f2)

