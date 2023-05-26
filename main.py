import argparse
import torch
from generate_3d_dataset import generate_dataset
from train_decoder import train_decoder
from train_triplane import train_triplane
from visualize import visualize


def main(args):

    raw_data_path = './dataset/' + args.shape_name + '.obj'
    data_path = './dataset/' + args.shape_name + '_sample.npy'
    triplane_path = './dataset/' + args.shape_name + '_triplane.npy'
    model_path = './model/'  + args.shape_name + '.pt'
    output_path = './output/' + args.shape_name + '.obj' 

    device = None
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    if not args.skip_data:
        print("-"*60)
        print("Preprocessing dataset")
        generate_dataset(raw_data_path, data_path, args.count, args.epsilon)
    
    if not args.skip_decoder:
        print("-"*60)
        print("Training decoder")
        train_decoder(data_path, model_path, args.decoder_epoches, device, args.epsilon)
     
    if not args.skip_triplane:
        print("-"*60)
        print("Training triplane")
        train_triplane(data_path, triplane_path, model_path, args.triplane_epoches, device, args.epsilon)
    
    print("-"*60)
    print("Decoding")
    visualize(triplane_path, output_path, model_path, args.res, device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--shape_name', type=str, required=True)
    # parser.add_argument('--raw_data', type=str, required=True)
    # parser.add_argument('--data', type=str, required=True)
    # parser.add_argument('--model_path', type=str, required=True)
    # parser.add_argument('--triplane_path', type=str, required=True)
    # parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--count', type=int, default=1000000)
    parser.add_argument('--epsilon', type=float, default=0.005)
    parser.add_argument('--res', type=int, default='256', required=False)
    parser.add_argument('--skip_data', type=bool, default=False)
    parser.add_argument('--skip_decoder', type=bool, default=False)
    parser.add_argument('--decoder_epoches', type=int, default=2000)
    parser.add_argument('--skip_triplane', type=bool, default=False)
    parser.add_argument('--triplane_epoches', type=int, default=400)   
    args = parser.parse_args()
    main(args)