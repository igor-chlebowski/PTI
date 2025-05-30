import os
import shutil
import argparse


def parse_filename(name):
    """Extract x, y, w, h from filename like 'x_y_w_h.ext'"""
    base, _ = os.path.splitext(name)
    parts = base.split('_')
    if len(parts) != 4:
        return None
    try:
        x, y, w, h = map(int, parts)
        return x, y, w, h
    except ValueError:
        return None


def intersection_area(box_a, box_b):
    """Compute intersection area between two boxes a and b."""
    xa, ya, wa, ha = box_a
    xb, yb, wb, hb = box_b
    x_left = max(xa, xb)
    y_top = max(ya, yb)
    x_right = min(xa + wa, xb + wb)
    y_bottom = min(ya + ha, yb + hb)
    if x_right <= x_left or y_bottom <= y_top:
        return 0
    return (x_right - x_left) * (y_bottom - y_top)


def build_tree(boxes, overlap_threshold):
    """
    Build parent-child tree such that each box is assigned to the
    single smallest-area parent where overlap/child_area >= threshold.
    Returns list of root indices and children mapping.
    """
    # Sort indices by descending area so larger boxes are considered first
    sorted_idxs = sorted(range(len(boxes)), key=lambda i: boxes[i]['area'], reverse=True)
    parent_map = {}
    children = {i: [] for i in sorted_idxs}

    for i in sorted_idxs:
        best_parent = None
        best_parent_area = None
        box_i = (boxes[i]['x'], boxes[i]['y'], boxes[i]['w'], boxes[i]['h'])
        # Find the smallest parent that satisfies overlap threshold
        for j in sorted_idxs:
            if boxes[j]['area'] <= boxes[i]['area']:
                continue
            box_j = (boxes[j]['x'], boxes[j]['y'], boxes[j]['w'], boxes[j]['h'])
            inter = intersection_area(box_j, box_i)
            if inter / boxes[i]['area'] >= overlap_threshold:
                # candidate parent
                if best_parent is None or boxes[j]['area'] < best_parent_area:
                    best_parent = j
                    best_parent_area = boxes[j]['area']
        parent_map[i] = best_parent
        if best_parent is not None:
            children[best_parent].append(i)

    # Roots = boxes with no parent
    roots = [i for i, p in parent_map.items() if p is None]
    return roots, children


def save_tree(idx, boxes, children, src_dir, dst_dir):
    """Recursively save nested folders and copy image files."""
    b = boxes[idx]
    folder = f"{b['x']}_{b['y']}_{b['w']}_{b['h']}"
    target_path = os.path.join(dst_dir, folder)
    os.makedirs(target_path, exist_ok=True)
    # Copy this symbol image
    src_file = os.path.join(src_dir, b['fname'])
    dst_file = os.path.join(target_path, b['fname'])
    shutil.copy2(src_file, dst_file)
    # Recurse to direct children only
    for child in children.get(idx, []):
        save_tree(child, boxes, children, src_dir, target_path)


def main():
    parser = argparse.ArgumentParser(
        description='Build nested structure of symbol images based on overlap ratio')
    parser.add_argument('src_dir', help='Directory with images named x_y_w_h.ext')
    parser.add_argument('dst_dir', help='Output directory for nested tree')
    parser.add_argument('--overlap-threshold', type=float, default=0.5,
                        help='Minimum fraction of child area overlapping parent (0-1)')
    args = parser.parse_args()

    # Read and parse all boxes
    boxes = []
    for fname in os.listdir(args.src_dir):
        parsed = parse_filename(fname)
        if parsed:
            x, y, w, h = parsed
            boxes.append({'fname': fname, 'x': x, 'y': y, 'w': w, 'h': h, 'area': w * h})

    # Build tree mapping
    roots, children = build_tree(boxes, args.overlap_threshold)

    # Save each root subtree
    for r in roots:
        save_tree(r, boxes, children, args.src_dir, args.dst_dir)

if __name__ == '__main__':
    main()
