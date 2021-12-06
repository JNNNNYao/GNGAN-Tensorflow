import os
import imageio
import argparse


def get_arguments():
    parser = argparse.ArgumentParser(description="GIF")
    parser.add_argument("--path", type=str, default="./logdir/sample", help="path to sample folder")
    return parser.parse_args()


def recursive_glob(rootdir=".", suffix=""):
    """Performs recursive glob with given suffix and rootdir 
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    return [
        os.path.join(looproot, filename)
        for looproot, _, filenames in os.walk(rootdir)
        for filename in filenames
        if filename.endswith(suffix)
    ]


if __name__ == '__main__':
    args = get_arguments()
    paths = recursive_glob(args.path, 'png')
    paths = sorted(paths)
    print("Found {} images".format(len(paths)))

    images = []
    for path in paths:
        images.append(imageio.imread(path))
    imageio.mimsave('./sample.gif', images, fps=20)


# python3 gif.py --path=logdir/sample/