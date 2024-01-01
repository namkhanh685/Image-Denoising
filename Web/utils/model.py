from models.network_swinir import SwinIR as net
from utils import utils_option as option
import argparse
import torch

def define_model():
    json_path="D:/Hoctap/Project/DoAn2/Code/Web/options/masked_denoising/input_mask_80_90.json"
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='masked_denoising')
    parser.add_argument('--scale', type=int, default=1, help='scale factor: 1, 2, 3, 4, 8') # 1 for dn and jpeg car
    parser.add_argument('--noise', type=int, default=15, help='noise level: 15, 25, 50')
    parser.add_argument('--jpeg', type=int, default=40, help='scale factor: 10, 20, 30, 40')
    parser.add_argument('--training_patch_size', type=int, default=128, help='patch size used in training SwinIR. '
                                       'Just used to differentiate two different settings in Table 2 of the paper. '
                                       'Images are NOT tested patch by patch.')
    parser.add_argument('--large_model', action='store_true', help='use large model, only provided for real image sr')
    parser.add_argument('--model_path', type=str,
                        default='D:/Hoctap/Project/DoAn2/Code/Web/models/input_mask_80_90.pth')
    parser.add_argument('--folder_lq', type=str, default=None, help='input low-quality test image folder')
    parser.add_argument('--folder_gt', type=str, default=None, help='input ground-truth test image folder')
    parser.add_argument('--tile', type=int, default=None, help='Tile size, None for no tile during testing (testing as a whole)')
    parser.add_argument('--tile_overlap', type=int, default=32, help='Overlapping of different tiles')
    parser.add_argument('--opt', type=str, default=json_path ,help='Path to option JSON file.')
    parser.add_argument('--name', type=str, default="test", help='Path to option JSON file.')
    opt = option.parse(parser.parse_args().opt, is_train=False)
    global opt_net
    opt_net = opt['netG']

    args = parser.parse_args()

    model = net(upscale=opt_net['upscale'],
                   in_chans=opt_net['in_chans'],
                   img_size=opt_net['img_size'],
                   window_size=opt_net['window_size'],
                   img_range=opt_net['img_range'],
                   depths=opt_net['depths'],
                   embed_dim=opt_net['embed_dim'],
                   num_heads=opt_net['num_heads'],
                   mlp_ratio=opt_net['mlp_ratio'],
                   upsampler=opt_net['upsampler'],
                   resi_connection=opt_net['resi_connection'],
                   talking_heads=opt_net['talking_heads'], 
                   use_attn_fn=opt_net['attn_fn'],                   
                   head_scale=opt_net['head_scale'],                   
                   on_attn=opt_net['on_attn'],     
                   use_mask=opt_net['use_mask'],     
                   mask_ratio1=opt_net['mask_ratio1'],     
                   mask_ratio2=opt_net['mask_ratio2'],     
                   mask_is_diff=opt_net['mask_is_diff'],     
                   type=opt_net['type'],     
                   opt=opt_net,     
                   )
    param_key_g = 'params'

    
    pretrained_model = torch.load(args.model_path)
    model.load_state_dict(pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model, strict=True)
        
    return model