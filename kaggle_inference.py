import argparse
import os
import sconf
import docker
import subprocess


def parse_cfg():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg", default='cfg/test/BirdCLEF.yaml', type=str,
    )
    parser.add_argument(
        '--docker_image', default='guts', type=str,
    )
    args, left_argv = parser.parse_known_args()
    cfg = sconf.Config(args.cfg)
    cfg.argv_update(left_argv)
    return cfg, args


def docker_inference():
    cfg, args = parse_cfg()
    print(cfg)
    mounted_folders = {
        os.path.dirname(cfg.checkpoint_path),
        os.path.dirname(cfg.datamodule_params.dataset_params.csv_path),
        os.path.dirname(cfg.datamodule_params.dataset_params.audio_folder),
        'codes',
        'cfg'
    }
    volume_mapping = ' '.join([f'-v {os.path.abspath(folder)}:{os.path.abspath(folder)}' for folder in mounted_folders])
    command = f"docker run \
        --gpus all \
        {volume_mapping} \
        {args.docker_image} \
        /bin/bash -c 'cd codes/BirdCLEF2023 && python codes/main.py --cfg {args.cfg}'"
    os.system(command)


if __name__ == "__main__":
    client = docker.from_env()
    image_tar_path = 'build/guts.tar'
    with open(image_tar_path, 'rb') as f:
        client.images.load(f.read())
    docker_inference()
