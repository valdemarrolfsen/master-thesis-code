import argparse
import os
import subprocess


def run(args):
    top = args.dir
    if top is None:
        top = '.'
        print('Using current directory')

    for folder in os.listdir(top):
        if not folder.startswith(args.folder_prefix):
            continue

        for file in os.listdir(os.path.join(top, folder)):
            # let's grab the .gdb file
            if not file.endswith('.gdb'):
                continue
            subprocess.run(["ogr2ogr", os.path.join(top, folder, 'shape'), os.path.join(top, folder, file)])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir',
                        type=str,
                        help='Directory to run in',
                        default=None
                        )
    parser.add_argument('--folder-prefix',
                        type=str,
                        help='Folder prefix for folders to convert files in',
                        required=True,
                        )
    args = parser.parse_args()
    run(args)
