import torchsummary
import torch
from torchvision.models import alexnet, squeezenet1_0, shufflenet_v2_x0_5, mobilenet_v2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = alexnet()

model.to(device)

model.load_state_dict(torch.load("../data/models/cnrpark/cnrpark_ext/80_20/alexnet.pth"))

torchsummary.summary(model, (3, 224, 224))
