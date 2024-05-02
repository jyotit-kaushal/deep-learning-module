from model.supernet_transformer import Vision_TransformerSuper
import torch
input_size=224
patch_size=16
                       
drop_path=0.1
max_relative_position = 14
nb_classes = 1000
drop=0.0
# Define device for torch
use_cuda = True
print("CUDA is available:", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_VIT = Vision_TransformerSuper(img_size=input_size,
                                patch_size=patch_size,
                                embed_dim=256, depth=14,
                                num_heads=4,mlp_ratio=4.0,
                                qkv_bias=True, drop_rate=drop,
                                drop_path_rate=drop_path,
                                gp=True,
                                num_classes=nb_classes,
                                max_relative_position=max_relative_position,
                                relative_position=True,
                                change_qkv=True, abs_pos=not True)


model_VIT.to(device)
model_without_ddp=model_VIT
