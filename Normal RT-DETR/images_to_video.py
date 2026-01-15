"""
Convert VisDrone VID image sequences to video
Also works for any folder of sequential images
"""

import cv2
import os
import glob
import argparse
from pathlib import Path


def images_to_video(image_folder, output_video, fps=30, pattern="*.jpg"):
    """
    Convert a folder of images to a video file.
    
    Args:
        image_folder: Path to folder containing images
        output_video: Output video file path
        fps: Frames per second (default: 30)
        pattern: Image file pattern (default: *.jpg)
    """
    # Get all images
    image_pattern = os.path.join(image_folder, pattern)
    images = sorted(glob.glob(image_pattern))
    
    if len(images) == 0:
        print(f"❌ No images found matching pattern: {image_pattern}")
        return
    
    print(f"Found {len(images)} images")
    print(f"First image: {images[0]}")
    print(f"Last image: {images[-1]}")
    
    # Read first image to get dimensions
    first_frame = cv2.imread(images[0])
    if first_frame is None:
        print(f"❌ Could not read first image: {images[0]}")
        return
    
    height, width, channels = first_frame.shape
    print(f"Video resolution: {width}x{height}")
    print(f"FPS: {fps}")
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    # Write all frames
    print(f"\nCreating video...")
    for i, image_path in enumerate(images):
        frame = cv2.imread(image_path)
        
        if frame is None:
            print(f"⚠  Warning: Could not read {image_path}, skipping...")
            continue
        
        out.write(frame)
        
        # Progress
        if (i + 1) % 10 == 0:
            progress = ((i + 1) / len(images)) * 100
            print(f"Progress: {progress:.1f}% ({i + 1}/{len(images)})", end='\r')
    
    out.release()
    
    duration = len(images) / fps
    print(f"\n\n✓ Video created successfully!")
    print(f"  Output: {output_video}")
    print(f"  Frames: {len(images)}")
    print(f"  Duration: {duration:.2f}s")


def convert_visdrone_sequences(visdrone_vid_folder, output_folder, fps=30):
    """
    Convert all VisDrone VID sequences to videos.
    
    VisDrone VID structure:
    visdrone_vid_folder/
    ├── uav0000013_00000_v/
    │   ├── 0000001.jpg
    │   ├── 0000002.jpg
    │   └── ...
    ├── uav0000013_01073_v/
    │   └── ...
    """
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    
    # Find all sequence folders
    sequence_folders = [f for f in glob.glob(os.path.join(visdrone_vid_folder, "*"))
                       if os.path.isdir(f)]
    
    if len(sequence_folders) == 0:
        print(f"❌ No sequence folders found in: {visdrone_vid_folder}")
        return
    
    print(f"Found {len(sequence_folders)} video sequences")
    
    for i, seq_folder in enumerate(sequence_folders):
        seq_name = os.path.basename(seq_folder)
        output_video = os.path.join(output_folder, f"{seq_name}.mp4")
        
        print(f"\n[{i+1}/{len(sequence_folders)}] Processing: {seq_name}")
        images_to_video(seq_folder, output_video, fps)
    
    print(f"\n{'='*60}")
    print(f"✓ All sequences converted!")
    print(f"  Output folder: {output_folder}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description='Convert image sequences to video')
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Input folder (single sequence or parent folder with multiple sequences)')
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='Output video file or folder')
    parser.add_argument('--fps', type=int, default=30,
                        help='Frames per second (default: 30)')
    parser.add_argument('--pattern', type=str, default='*.jpg',
                        help='Image file pattern (default: *.jpg)')
    parser.add_argument('--visdrone-mode', action='store_true',
                        help='Process multiple VisDrone VID sequences (input should be parent folder)')
    
    args = parser.parse_args()
    
    if args.visdrone_mode:
        # Process multiple sequences
        convert_visdrone_sequences(args.input, args.output, args.fps)
    else:
        # Process single sequence
        images_to_video(args.input, args.output, args.fps, args.pattern)


if __name__ == "__main__":
    main()
