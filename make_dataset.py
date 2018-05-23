import argparse
import os

from preprocessing import label_generator

datasets = [
    'drammen/tiled-512x512/',
]


def run():
    # Set ut the argument parser
    ap = argparse.ArgumentParser()
    ap.add_argument('--input-base', type=str, required=True, help='path to input file')
    ap.add_argument('--output', type=str, required=True, help='path for output file')
    ap.add_argument('--classname', type=str, required=True, help='classname')
    args = ap.parse_args()
    for dataset in datasets:
        dataset = os.path.join(args.input_base, dataset)
        a = {'output': args.output,
             'input': dataset,
             'color': 1 if args.classname != 'multiclass' else 'color',
             'prefix': '',
             'include_empty': False,
             'binary': True if args.classname != 'multiclass' else False,
             'res': 512,
             'class_name': args.classname,
             'threads': 8}
        label_generator.run(a)


if __name__ == '__main__':
    run()
