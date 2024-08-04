import argparse
import subprocess
import os
import os.path as osp
import glob
import tqdm

def split_video_to_frames(video_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    ffmpeg_command = [
        'ffmpeg',
        '-i', video_path,
        '-hide_banner', '-loglevel', 'error',  # Quiet the output
        os.path.join(output_folder, '%06d.png')
    ]

    subprocess.run(ffmpeg_command, check=True)

def main(args):

    root_dir = args.root_dir
    seq_name = args.seq_name

    if seq_name != 'all':
        video_paths = sorted(glob.glob(osp.join(root_dir, seq_name, '*/*.mp4')))
    else:
        video_paths = sorted(glob.glob(osp.join(root_dir, '*/*/*.mp4')))
    
    # split each video file into frames
    for video_path in tqdm.tqdm(video_paths):
        stem, _ = osp.splitext(osp.basename(video_path))
        output_folder = osp.join(osp.dirname(video_path), stem)
        split_video_to_frames(video_path, output_folder)

    print('All videos have been split into frames.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, required=True,
                        help='root directory in which data is stored.')
    parser.add_argument('--seq_name', type=str, default='all',
                        help='sequence name to process. all sequences will be processed if not specified.')
    args = parser.parse_args()

    main(args)
