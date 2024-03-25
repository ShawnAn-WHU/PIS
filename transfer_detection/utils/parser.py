import argparse as ag
import json

def get_parser_with_args(metadata_json='metadata.json'):
    parser = ag.ArgumentParser(description='Training change detection network')

    with open(metadata_json, 'r') as fin:
        metadata = json.load(fin)
        parser.set_defaults(**metadata)
        

    parser.add_argument('--backbone', default='resnet', type=str, choices=['resnet','swin','vitae'], help='type of model')

    parser.add_argument('--dataset', default='cdd', type=str, choices=['cdd','levir'], help='type of dataset')

    parser.add_argument('--mode', default='emp', type=str, choices=['spring', '4variants', 'gersp', 'emp_fix400', 'imagenet', 'random', 'swin_b', 'moco', 'tov', 'imp','rsp_40', 'rsp_100', 'rsp_120' , 'rsp_300', 'rsp_300_sgd', 'seco', 'emp'], help='type of pretrn')

    parser.add_argument('--path', default='./', type=str, help='path of saved model')


    return parser, metadata

