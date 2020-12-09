from models import *
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Darknet("/content/Rolo3/yolo3/config/yolov3.cfg", img_size=416).to(device)
model.load_state_dict(torch.load("/content/yolo3/yolo3.pth"))
i = 106
##################################################################################
i+=1
route = nn.Sequential()
route.add_module(f"route_{i}", EmptyLayer())
model.module_list.append(route)

i+=1
cnn = nn.Sequential()
cnn.add_module(
                f"conv_{i}",
                nn.Conv2d(
                    in_channels=512,
                    out_channels=64,
                    kernel_size=1,
                    stride=1,
                    padding=1,
                    bias=0,
                ))
cnn.add_module(f"batch_norm_{i}", nn.BatchNorm2d(64, momentum=0.9, eps=1e-5))
cnn.add_module(f"leaky_{i}", nn.LeakyReLU(0.1))
model.module_list.append(cnn)

i+=1
route = nn.Sequential()
route.add_module(f"route_{i}", EmptyLayer())
model.module_list.append(route)

i+=1
cnn = nn.Sequential()
cnn.add_module(
                f"conv_{i}",
                nn.Conv2d(
                    in_channels=255,
                    out_channels=15,
                    kernel_size=1,
                    stride=1,
                    padding=1,
                    bias=1,
                ))
model.module_list.append(cnn)

i+=1
flatten = nn.Sequential()
flatten.add_module(f"flatten_{i}", EmptyLayer())
model.module_list.append(flatten)

i+=1
regression = nn.Sequential()
regression.add_module(
            f"regression_{i}",
            nn.Linear(13351, 1024),
            )
regression.add_module(
            f"relu_{i}",
            nn.ReLU(inplace=True)
            )
model.module_list.append(regression)

i+=1
lstm = nn.Sequential()
lstm.add_module(
            f"lstm_{i}",
            nn.LSTM(input_size=1024, hidden_size=512, batch_first=True),
            )
model.module_list.append(lstm)

i+=1
regression = nn.Sequential()
regression.add_module(
            f"regression_{i}",
            nn.Linear(512, 2535),
            )
regression.add_module(
            f"relu_{i}",
            nn.ReLU(inplace=True)
            )
model.module_list.append(regression)
############################################################################

i+=1
route = nn.Sequential()
route.add_module(f"route_{i}", EmptyLayer())
model.module_list.append(route)

i+=1
cnn = nn.Sequential()
cnn.add_module(
                f"conv_{i}",
                nn.Conv2d(
                    in_channels=256,
                    out_channels=32,
                    kernel_size=1,
                    stride=1,
                    padding=1,
                    bias=0,
                ))
cnn.add_module(f"batch_norm_{i}", nn.BatchNorm2d(32, momentum=0.9, eps=1e-5))
cnn.add_module(f"leaky_{i}", nn.LeakyReLU(0.1))
model.module_list.append(cnn)

i+=1
route = nn.Sequential()
route.add_module(f"route_{i}", EmptyLayer())
model.module_list.append(route)

i+=1
cnn = nn.Sequential()
cnn.add_module(
                f"conv_{i}",
                nn.Conv2d(
                    in_channels=255,
                    out_channels=15,
                    kernel_size=1,
                    stride=1,
                    padding=1,
                    bias=1,
                ))
model.module_list.append(cnn)

i+=1
flatten = nn.Sequential()
flatten.add_module(f"flatten_{i}", EmptyLayer())
model.module_list.append(flatten)

i+=1
regression = nn.Sequential()
regression.add_module(
            f"regression_{i}",
            nn.Linear(31772, 1024),
            )
regression.add_module(
            f"relu_{i}",
            nn.ReLU(inplace=True)
            )
model.module_list.append(regression)

i+=1
lstm = nn.Sequential()
lstm.add_module(
            f"lstm_{i}",
            nn.LSTM(input_size=1024, hidden_size=512, batch_first=True),
            )
model.module_list.append(lstm)

i+=1
regression = nn.Sequential()
regression.add_module(
            f"regression_{i}",
            nn.Linear(512, 10140),
            )
regression.add_module(
            f"relu_{i}",
            nn.ReLU(inplace=True)
            )
model.module_list.append(regression)
###########################################################
i+=1
route = nn.Sequential()
route.add_module(f"route_{i}", EmptyLayer())
model.module_list.append(route)

i+=1
cnn = nn.Sequential()
cnn.add_module(
                f"conv_{i}",
                nn.Conv2d(
                    in_channels=128,
                    out_channels=16,
                    kernel_size=1,
                    stride=1,
                    padding=1,
                    bias=0,
                ))
cnn.add_module(f"batch_norm_{i}", nn.BatchNorm2d(16, momentum=0.9, eps=1e-5))
cnn.add_module(f"leaky_{i}", nn.LeakyReLU(0.1))
model.module_list.append(cnn)

i+=1
route = nn.Sequential()
route.add_module(f"route_{i}", EmptyLayer())
model.module_list.append(route)

i+=1
cnn = nn.Sequential()
cnn.add_module(
                f"conv_{i}",
                nn.Conv2d(
                    in_channels=255,
                    out_channels=15,
                    kernel_size=1,
                    stride=1,
                    padding=1,
                    bias=1,
                ))
model.module_list.append(cnn)

i+=1
flatten = nn.Sequential()
flatten.add_module(f"flatten_{i}", EmptyLayer())
model.module_list.append(flatten)

i+=1
regression = nn.Sequential()
regression.add_module(
            f"regression_{i}",
            nn.Linear(83824, 1024),
            )
regression.add_module(
            f"relu_{i}",
            nn.ReLU(inplace=True)
            )
model.module_list.append(regression)

i+=1
lstm = nn.Sequential()
lstm.add_module(
            f"lstm_{i}",
            nn.LSTM(input_size=1024, hidden_size=512, batch_first=True),
            )
model.module_list.append(lstm)

i+=1
regression = nn.Sequential()
regression.add_module(
            f"regression_{i}",
            nn.Linear(512, 40560),
            )
regression.add_module(
            f"relu_{i}",
            nn.ReLU(inplace=True)
            )
model.module_list.append(regression)

i+=1
rolo = nn.Sequential()
rolo.add_module(f"rolo_{i}", EmptyLayer())
model.module_list.append(rolo)

torch.save(model.state_dict(),'rolo3.pth')
print(model)