import json
import argparse
from pathlib import Path
import pandas as pd
from PIL import Image
import xml.etree.ElementTree as ET
from xml.dom import minidom

# Supported image extensions
IMAGE_EXTS = [".tif"]


def prettify_xml(elem):
    """Return pretty-formatted XML string."""
    rough_string = ET.tostring(elem, encoding="utf-8")
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")


def create_voc_xml(filename, folder, width, height, depth, class_name, output_path):
    """
    Create one Pascal VOC XML file for an image.
    Since this is classification data, we use the whole image as a proxy box.
    """
    annotation = ET.Element("annotation")

    ET.SubElement(annotation, "folder").text = folder
    ET.SubElement(annotation, "filename").text = filename

    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = str(depth)

    ET.SubElement(annotation, "segmented").text = "0"

    obj = ET.SubElement(annotation, "object")
    ET.SubElement(obj, "name").text = str(class_name)
    ET.SubElement(obj, "pose").text = "Unspecified"
    ET.SubElement(obj, "truncated").text = "0"
    ET.SubElement(obj, "difficult").text = "0"

    # Whole-image bounding box
    bndbox = ET.SubElement(obj, "bndbox")
    ET.SubElement(bndbox, "xmin").text = "1"
    ET.SubElement(bndbox, "ymin").text = "1"
    ET.SubElement(bndbox, "xmax").text = str(width)
    ET.SubElement(bndbox, "ymax").text = str(height)

    xml_str = prettify_xml(annotation)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(xml_str)


def normalize_columns(df):
    """
    Normalize column names so small differences are tolerated.
    Expected logical columns:
      - patch_name
      - ship type
    """
    normalized_map = {}
    for c in df.columns:
        normalized = c.strip().lower().replace("_", " ")
        normalized_map[normalized] = c

    patch_col = normalized_map.get("patch name")
    ship_col = normalized_map.get("ship type")

    if patch_col is None or ship_col is None:
        raise ValueError(
            f"CSV must contain columns equivalent to 'patch_name' and 'ship type'. "
            f"Found columns: {list(df.columns)}"
        )

    return patch_col, ship_col


def find_image_for_patch(csv_dir: Path, patch_name: str):
    """
    Find the image file corresponding to patch_name.

    Handles cases where:
    - patch_name already has extension
    - patch_name has no extension
    - images are in same folder or subfolders
    """
    patch_path = Path(str(patch_name).strip())

    # Case 1: patch_name already includes extension
    if patch_path.suffix:
        candidate = csv_dir / patch_path.name
        if candidate.exists():
            return candidate

        matches = list(csv_dir.rglob(patch_path.name))
        if matches:
            return matches[0]

    # Case 2: patch_name has no extension
    stem = patch_path.stem if patch_path.suffix else patch_path.name
    for ext in IMAGE_EXTS:
        candidate = csv_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate

    for ext in IMAGE_EXTS:
        matches = list(csv_dir.rglob(f"{stem}{ext}"))
        if matches:
            return matches[0]

    return None


def convert_dataset(root_dir, output_dir):
    root_dir = Path(root_dir)
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    voc_dir = output_dir / "voc_xml"
    voc_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # STEP 1: Find all ais.csv files
    # ------------------------------------------------------------------
    csv_files = list(root_dir.rglob("ais.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No ais.csv files found under: {root_dir}")

    print(f"Found {len(csv_files)} ais.csv file(s).")

    # ------------------------------------------------------------------
    # STEP 2: Read every CSV and collect rows
    # THIS IS THE PART WHERE THE CSV IS OPENED
    # ------------------------------------------------------------------
    all_rows = []
    class_names = set()

    for csv_file in csv_files:
        print(f"Reading CSV: {csv_file}")

        # OPEN THE CSV HERE
        df = pd.read_csv(csv_file)

        # Identify the correct column names
        patch_col, ship_col = normalize_columns(df)

        # Read each row from the CSV
        for _, row in df.iterrows():
            patch_name = str(row[patch_col]).strip()
            ship_type = str(row[ship_col]).strip()

            if not patch_name or patch_name.lower() == "nan":
                continue
            if not ship_type or ship_type.lower() == "nan":
                continue

            all_rows.append({
                "csv_file": csv_file,
                "patch_name": patch_name,
                "ship_type": ship_type
            })
            class_names.add(ship_type)

    # ------------------------------------------------------------------
    # STEP 3: Build category mapping
    # ------------------------------------------------------------------
    class_names = sorted(class_names)
    class_to_id = {name: i + 1 for i, name in enumerate(class_names)}

    coco = {
        "info": {
            "description": "Classification dataset exported into COCO-style JSON from ais.csv files",
            "version": "1.0",
            "year": 2026
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [
            {
                "id": class_to_id[name],
                "name": name,
                "supercategory": "ship"
            }
            for name in class_names
        ]
    }

    image_id = 1
    annotation_id = 1
    missing_images = []

    # ------------------------------------------------------------------
    # STEP 4: Match each CSV row to an image and create annotations
    # ------------------------------------------------------------------
    for item in all_rows:
        csv_file = item["csv_file"]
        patch_name = item["patch_name"]
        ship_type = item["ship_type"]

        csv_dir = csv_file.parent
        image_path = find_image_for_patch(csv_dir, patch_name)

        if image_path is None:
            missing_images.append((str(csv_file), patch_name))
            continue

        try:
            with Image.open(image_path) as img:
                width, height = img.size
                depth = 1 if img.mode == "L" else 3
        except Exception as e:
            print(f"[WARN] Cannot open image {image_path}: {e}")
            continue

        rel_image_path = image_path.relative_to(root_dir).as_posix()
        category_id = class_to_id[ship_type]

        # COCO image entry
        coco["images"].append({
            "id": image_id,
            "file_name": rel_image_path,
            "width": width,
            "height": height
        })

        # COCO annotation entry
        # Whole image used as proxy box
        coco["annotations"].append({
            "id": annotation_id,
            "image_id": image_id,
            "category_id": category_id,
            "bbox": [0, 0, width, height],
            "area": width * height,
            "iscrowd": 0,
            "segmentation": []
        })

        # Pascal VOC XML
        xml_subdir = voc_dir / image_path.parent.relative_to(root_dir)
        xml_subdir.mkdir(parents=True, exist_ok=True)
        xml_path = xml_subdir / f"{image_path.stem}.xml"

        create_voc_xml(
            filename=image_path.name,
            folder=image_path.parent.name,
            width=width,
            height=height,
            depth=depth,
            class_name=ship_type,
            output_path=xml_path
        )

        image_id += 1
        annotation_id += 1

    # ------------------------------------------------------------------
    # STEP 5: Save outputs
    # ------------------------------------------------------------------
    coco_path = output_dir / "annotations_coco.json"
    with open(coco_path, "w", encoding="utf-8") as f:
        json.dump(coco, f, indent=2)

    missing_path = output_dir / "missing_images.txt"
    with open(missing_path, "w", encoding="utf-8") as f:
        for csv_name, patch_name in missing_images:
            f.write(f"{csv_name} -> {patch_name}\n")

    print("\nConversion finished.")
    print(f"COCO JSON saved to: {coco_path}")
    print(f"VOC XML files saved under: {voc_dir}")
    print(f"Missing image report: {missing_path}")
    print(f"Total categories: {len(class_to_id)}")
    print(f"Total converted images: {len(coco['images'])}")
    print(f"Missing images: {len(missing_images)}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert a classification dataset with many ais.csv files into COCO JSON and Pascal VOC XML."
    )
    parser.add_argument(
        "--root_dir",
        required=True,
        help="Root directory containing many folders, each with ais.csv and image patches"
    )
    parser.add_argument(
        "--output_dir",
        default="converted_annotations",
        help="Directory to save COCO JSON and Pascal VOC XML files"
    )

    args = parser.parse_args()
    convert_dataset(args.root_dir, args.output_dir)


if __name__ == "__main__":
    main()