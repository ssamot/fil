import torch
import torch.nn as nn
import torch.nn.functional as F


class Embedder(nn.Module):
    def __init__(self, num_categories_list):
        super().__init__()
        self.num_categories_list = num_categories_list
        self.convert_layers = nn.ModuleList([
            nn.Linear(num_categories, 1) for num_categories in num_categories_list
        ])

    def forward(self, inputs):
        """
        inputs: list of LongTensor with categorical indices,
                each shape: (batch_size,)
        Returns:
          List of tensors (batch_size,) — one scalar per input per batch element
        """
        outputs = []
        for i, (input_cat, linear_layer) in enumerate(zip(inputs, self.convert_layers)):
            one_hot = F.one_hot(input_cat, num_classes=self.num_categories_list[i]).float()
            scalar = linear_layer(one_hot).squeeze(-1)  # shape (batch_size,)
            outputs.append(scalar)
        return outputs


# Example usage
batch_size = 3
num_categories_list = [5, 10, 3]

model = Embedder(num_categories_list)

inputs = [
    torch.tensor([1, 0, 4]),
    torch.tensor([3, 9, 0]),
    torch.tensor([2, 1, 2])
]

outputs = model(inputs)
for i, out in enumerate(outputs):
    print(f"Output {i}:", out)
