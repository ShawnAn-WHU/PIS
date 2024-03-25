import argparse as ag
import json

def get_parser_with_args(metadata_json='metadata.json'):
    parser = ag.ArgumentParser(description='Training change detection network')

    with open(metadata_json, 'r') as fin:
        metadata = json.load(fin)
        parser.set_defaults(**metadata)
        

    parser.add_argument('--backbone', default='resnet', type=str, choices=['resnet','swin'], help='type of model')

    parser.add_argument('--dataset', default='cdd', type=str, choices=['cdd','levir'], help='type of dataset')

    parser.add_argument('--mode', default='emp', type=str, choices=['random', 'imagenet', 'seco', 'moco', 'tov', 'gersp', 'pis-r50', 'pis-swinb'], help='type of pretrn')

    parser.add_argument('--path', default='./', type=str, help='path of saved model')


    return parser, metadata

