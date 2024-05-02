
from model.subnet_transformer import Vision_TransformerSubnet

input_size=224
patch_size=16
                       
drop_path=0.1
max_relative_position = 14
nb_classes = 1000
drop=0.0

def get_subnet_arch(config):
    model = Vision_TransformerSubnet(img_size=input_size,
                                patch_size=patch_size,
                                embed_dim=config['embed_dim'],
                                depth=config['layer_num'],
                                num_heads=config['num_heads'],
                                mlp_ratio=config['mlp_ratio'],
                                qkv_bias=True, drop_rate=drop,
                                drop_path_rate=drop_path,
                                gp=True,
                                num_classes=nb_classes,
                                max_relative_position=max_relative_position,
                                relative_position=True,
                                change_qkv=True, abs_pos=not True)
    return model
