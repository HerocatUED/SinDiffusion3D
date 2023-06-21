import argparse
import torch
from generate_3d_dataset import generate_dataset
from train import train
from finetune_mlp import finetune
from visualize import visualize
from utils import triplane2img


def main(args):

    shape_list = ['Oak_Tree', 'Palm_Tree', 'cactus', 'StoneWall', 'Rock', 'Table']
    if args.shape_name is not None:
        shape_list = [args.shape_name]
    for shape_name in shape_list:
        print(f"Processing {shape_name}")

        raw_data_path = './dataset/' + shape_name + '.obj'
        data_path = './dataset/' + shape_name + '_sample.npy'
        triplane_path = './dataset/' + shape_name + '_triplane.npy'
        model_path = './model/'  + shape_name
        output_path = './output/' + shape_name + '.obj' 
        triimg_path = './output/' + shape_name + '_triimg.png'

        device = None
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        
        if not args.skip_data:
            print("-"*80)
            generate_dataset(raw_data_path, data_path, args.count, args.epsilon)
        
        if not args.skip_train:
            print("-"*80)
            print("Training")
            train(data_path, model_path, triplane_path, args.train_epoches, device, args.epsilon)
        
        if not args.skip_finetune:
            print("-"*80)
            print("Finetuning MLP")
            finetune(data_path, model_path, args.finetune_epoches, device, args.epsilon)
        
        print("-"*80)
        print("Decoding")
        visualize(triplane_path, output_path, model_path+'_mlp.pt', args.resolution, device)

        triplane2img(triplane_path, triimg_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--shape_name', type=str, default=None)
    parser.add_argument('--count', type=int, default=1000000)
    parser.add_argument('--epsilon', type=float, default=0.01)
    parser.add_argument('--resolution', type=int, default='256')
    parser.add_argument('--skip_data', type=bool, default=False)
    parser.add_argument('--skip_train', type=bool, default=False)
    parser.add_argument('--train_epoches', type=int, default=3000)
    parser.add_argument('--skip_finetune', type=bool, default=False)
    parser.add_argument('--finetune_epoches', type=int, default=100)   
    args = parser.parse_args()

    main(args)