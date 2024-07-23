import os
import ast
from argparse import ArgumentParser
from mmengine.config import Config
from mmdet.apis import DetInferencer
from mmdet.evaluation import get_classes

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('inputs', type=str, help='Input image file or folder path.')
    parser.add_argument('model', type=str, help='Config or checkpoint .pth file or the model name and alias defined in metafile. The model configuration file will try to read from .pth if the parameter is a .pth weights file.')
    parser.add_argument('--weights', default=None, help='Checkpoint file')
    parser.add_argument('--out-dir', type=str, default='outputs', help='Output directory of images or prediction results.')
    parser.add_argument('--texts', help='text prompt, such as "bench . car .", "$: coco"')
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    parser.add_argument('--pred-score-thr', type=float, default=0.3, help='bbox score threshold')
    parser.add_argument('--batch-size', type=int, default=1, help='Inference batch size.')
    parser.add_argument('--show', action='store_true', help='Display the image in a popup window.')
    parser.add_argument('--no-save-vis', action='store_true', help='Do not save detection vis results')
    parser.add_argument('--no-save-pred', action='store_true', help='Do not save detection json results')
    parser.add_argument('--print-result', action='store_true', help='Whether to print the results.')
    parser.add_argument('--palette', default='none', choices=['coco', 'voc', 'citys', 'random', 'none'], help='Color palette used for visualization')
    parser.add_argument('--custom-entities', '-c', action='store_true', help='Whether to customize entity names? If so, the input text should be "cls_name1 . cls_name2 . cls_name3 ." format')
    parser.add_argument('--chunked-size', '-s', type=int, default=-1, help='If the number of categories is very large, you can specify this parameter to truncate multiple predictions.')
    parser.add_argument('--tokens-positive', '-p', type=str, help='Used to specify which locations in the input text are of interest to the user. -1 indicates that no area is of interest, None indicates ignoring this parameter. The two-dimensional array represents the start and end positions.')
    parser.add_argument('--print-config', action='store_true', help='Whether to print the merged configuration.')
    return parser.parse_args()

def main():
    args = parse_args()

    # Load and merge configuration
    cfg = Config.fromfile(args.model)
    if args.weights and args.weights.endswith('.pth'):
        cfg.model.pretrained = args.weights
    
    # Print merged configuration if requested
    if args.print_config:
        print(f'Using configuration:\n{cfg.pretty_text}')

    # Determine if input is a directory or a single file
    if os.path.isdir(args.inputs):
        image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
        image_files = [os.path.join(args.inputs, f) for f in os.listdir(args.inputs) if f.lower().endswith(image_extensions)]
    else:
        image_files = [args.inputs]

    # Initialize the inferencer
    init_args = {
        'model': args.model,
        'weights': args.weights,
        'device': args.device,
        'palette': args.palette
    }
    inferencer = DetInferencer(**init_args)

    for image_file in image_files:
        print(f'Processing {image_file}')
        result = inferencer(image_file, batch_size=args.batch_size, pred_score_thr=args.pred_score_thr)

        if args.show:
            inferencer.visualize(image_file, result, show=True)

        if args.out_dir:
            os.makedirs(args.out_dir, exist_ok=True)
            output_file = os.path.join(args.out_dir, os.path.basename(image_file))
            inferencer.visualize(image_file, result, out_file=output_file)

        if args.print_result:
            print(result)

if __name__ == '__main__':
    main()
