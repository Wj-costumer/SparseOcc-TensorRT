import torch

import torch.nn.functional as F


class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, m):
        return F.grid_sample(x, m)


x = torch.rand(1, 1, 10, 10)
m = torch.Tensor([[[1, 0, 0], [0, 1, 0]]])
m = F.affine_grid(m, x.size()).type_as(x)

model = MyModel()

torch.onnx.export(
    model,
    (x, m),
    "grid_sample.onnx",
    verbose=True,
    opset_version=17,
    input_names=['input', 'grid'],
    output_names=['output']
)