
import argparse
import random
import shutil
import os

from Crawler.main import Crawler
from Segregator.process_image import prepare_image_for_contours, find_and_draw_contours, save_rois
from Segregator.segregator import parse_filename, build_tree, save_tree

def parse_args():
    parser = argparse.ArgumentParser(description="Process an image to find and save contours.")
    parser.add_argument("image_path", type=str, help="Path to the input image file.")
    parser.add_argument("output_dir", type=str, default="res", help="Directory to save the output images and ROIs.")
    parser.add_argument("--min_contour_area", type=int, default=20, help="Minimum area of contours to consider.")
    parser.add_argument("--padding_pixels", type=int, default=9, help="Padding in pixels to apply around detected contours.")
    parser.add_argument('--overlap-threshold', type=float, default=0.5,
                        help='Minimum fraction of child area overlapping parent (0-1)')
    global MIN_CONTOUR_AREA, PADDING_PIXELS
    MIN_CONTOUR_AREA = parser.parse_args().min_contour_area
    PADDING_PIXELS = parser.parse_args().padding_pixels
    return parser.parse_args()

def create_file_structure(args, src_dir, dest_dir):
    """
    Create a file structure based on the provided arguments.
    This function is a placeholder for the actual implementation.
    """
    # Here you would implement the logic to create the file structure
    # based on the source and destination directories.
    print(f"Creating file structure from {src_dir} to {dest_dir} with args: {args}")

    boxes = []
    for fname in os.listdir(src_dir):
        parsed = parse_filename(fname)
        if parsed:
            x, y, w, h = parsed
            boxes.append({'fname': fname, 'x': x, 'y': y, 'w': w, 'h': h, 'area': w * h})

    # Build tree mapping
    roots, children = build_tree(boxes, args.overlap_threshold)

    # Save each root subtree
    for r in roots:
        save_tree(r, boxes, children, src_dir, dest_dir)


def main():
    # Main function to execute the image processing workflow

    # Parse arguments
    args = parse_args()
    image_path = args.image_path
    temp_dir = "temp" + random.randint(0, 1000).__str__()
    output_dir = args.output_dir

    # Run the image processing function

    original, binary_after_thresh, closed_binary = prepare_image_for_contours(image_path)
    if original is not None and closed_binary is not None:
        image_with_boxes, padded_boxes = find_and_draw_contours(
            original, closed_binary, MIN_CONTOUR_AREA, PADDING_PIXELS
        )
        
        save_rois(original, padded_boxes, temp_dir)
    else:
        print("Error processing the image. Please check the input file path and format.")
        return


    # Run the file structure creation function
    create_file_structure(args, temp_dir, output_dir)

    # Clean up temporary directory
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)

    print(f"Processing complete. Tree structure saved in {output_dir}.")

    # Run file crawler function
    crawler = Crawler(output_dir)
    structure = crawler.crawl()

    print("Crawled file structure:")
    
    stack = [structure]
    while stack:
        current = stack.pop()
        print(f"{len(stack)}  {current['name']} ({current['type']}) - {current.get('detection', 'N/A')}")
        if 'children' in current:
            stack.extend(current['children'])

if __name__ == "__main__":
    main()