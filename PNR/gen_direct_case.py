import argparse
import pandas as pd
import os

from your_draw_module import draw

def add_entry(file_path, node_id, node_type, width, height, lx, ly, pin_count, cluster_area, internal_net, group_id, region_id):
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
    else:
        df = pd.DataFrame(columns=["node_id", "node_type", "width", "height", "lx", "ly", "pin_count", "cluster_area", "internal_net", "group_id", "region_id"])

    # Add the new entry
    new_entry = {
        "node_id": node_id,
        "node_type": node_type,
        "width": width,
        "height": height,
        "lx": lx,
        "ly": ly,
        "pin_count": pin_count,
        "cluster_area": cluster_area,
        "internal_net": internal_net,
        "group_id": group_id,
        "region_id": region_id
    }
    df = df.append(new_entry, ignore_index=True)

    df.to_csv(file_path, index=False)

    draw()

def main():
    parser = argparse.ArgumentParser(description="Populate a CSV file with node data and display it.")
    parser.add_argument("file_path", type=str, help="Path to the CSV file.")
    parser.add_argument("node_id", type=int, help="Node ID.")
    parser.add_argument("node_type", type=int, help="Node type.")
    parser.add_argument("width", type=int, help="Width.")
    parser.add_argument("height", type=int, help="Height.")
    parser.add_argument("lx", type=float, help="LX coordinate.")
    parser.add_argument("ly", type=float, help="LY coordinate.")
    parser.add_argument("pin_count", type=int, help="Pin count.")
    parser.add_argument("cluster_area", type=int, help="Cluster area.")
    parser.add_argument("internal_net", type=int, help="Internal net.")
    parser.add_argument("group_id", type=int, help="Group ID.")
    parser.add_argument("region_id", type=int, help="Region ID.")

    args = parser.parse_args()

    add_entry(args.file_path, args.node_id, args.node_type, args.width, args.height, args.lx, args.ly, args.pin_count, args.cluster_area, args.internal_net, args.group_id, args.region_id)

if __name__ == "__main__":
    main()