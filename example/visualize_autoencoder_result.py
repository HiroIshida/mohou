import argparse
from typing import Type
from mohou.types import ImageBase, get_element_type
from mohou.script_utils import visualize_image_reconstruction

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-pn', type=str, default='kuka_reaching', help='project name')
    parser.add_argument('-n', type=int, default=5, help='number of visualization')
    parser.add_argument('-image', type=str, default='RGBImage', help='image type')
    args = parser.parse_args()
    project_name = args.pn
    n_vis = args.n
    image_type: Type[ImageBase] = get_element_type(args.image)  # type: ignore
    visualize_image_reconstruction(project_name, image_type, n_vis)
