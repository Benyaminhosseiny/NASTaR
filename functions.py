def get_file_list(directory):
    import os

    tif_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.tif'):         
                full_path = os.path.join(root, file)
                # Skip if "Extracted_Kelvin_Waves" is in the path
                if "Extracted" not in full_path:
                    tif_files.append(full_path)


    # tif_files = [f for f in tif_files if not os.path.basename(f).startswith('QL')] # Exclude files starting with 'QL' [These are low resolution files only for visualisation]
    # tif_files = [f for f in tif_files if os.path.basename(f).startswith('image_HH.tif')] # Keep only files starting with 'image_HH.tif' [These are the high resolution files for processing]
    tif_files = [f for f in tif_files if os.path.basename(f) == 'image_HH.tif'] # Keep only files 'image_HH.tif' [These are the high resolution files for processing]
    
    # print(f"Found {len(tif_files)} tif files.")
    
    return tif_files

# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def geo_info_from_metadata(metadata_path):
    # Load metadata to generate geo tranformation using tiepoints and sampling info

    import xml.etree.ElementTree as ET
    import rasterio as rio

    tree = ET.parse(metadata_path)
    root = tree.getroot()

    # Find the "geographicInformation" element in the XML metadata

    geo_info_elem = root.find(".//geographicInformation")

    # Extract the text of the "EllipsoidName" element from geo_info_elem
    ellipsoid_elem = geo_info_elem.find(".//EllipsoidName")

    geo_info = {}

    ellipsoid = ellipsoid_elem.text if ellipsoid_elem is not None else None
    geo_info['ellipsoid'] = ellipsoid
    print("Ellipsoid Name:", ellipsoid)

    # Extracting SampledLineSpacing and SampledPixelSpacing from Image_Attributes

    # Find the "Image_Attributes" element in the XML metadata
    im_attr_info_elem = root.find(".//Image_Attributes")

    # Extract SampledLineSpacing
    elem = im_attr_info_elem.find(".//SampledLineSpacing")
    gsd_line = elem.text if elem is not None else None
    geo_info['gsd_line'] = gsd_line
    print("Sampled Line Spacing:", gsd_line)

    # Extract SampledPixelSpacing
    elem = im_attr_info_elem.find(".//SampledPixelSpacing")
    gsd_pixel = elem.text if elem is not None else None
    geo_info['gsd_pixel'] = gsd_pixel
    print("Sampled Pixel Spacing:", gsd_pixel)

    elem = geo_info_elem.find(".//SemiMajorAxis")
    SemiMajorAxis_name = elem.text if elem is not None else None
    geo_info['SemiMajorAxis'] = SemiMajorAxis_name
    print("SemiMajorAxis:", SemiMajorAxis_name)

    elem = geo_info_elem.find(".//SemiMinorAxis")
    SemiMinorAxis_name = elem.text if elem is not None else None
    geo_info['SemiMinorAxis'] = SemiMinorAxis_name
    print("SemiMinorAxis:", SemiMinorAxis_name)

    # Extract tie point information from the "geographicInformation" element

    lat_all = []
    lon_all = []
    tiepoints = []
    for tp in geo_info_elem.findall(".//TiePoint"):
        line = tp.findtext("Line")
        pixel = tp.findtext("Pixel")
        lat = tp.findtext("Latitude")
        lon = tp.findtext("Longitude")
        height = tp.findtext("Height")
        tiepoints.append( rio.control.GroundControlPoint( row=int(line), col=int(pixel), x=float(lon), y=float(lat)) )

        lat_all.append(float(lat))
        lon_all.append(float(lon))
    geo_info['tiepoints'] = tiepoints
    # lat_min = min(lat_all)
    # lon_min = min(lon_all)
    # lat_max = max(lat_all)
    # lon_max = max(lon_all)
    # print(f"Lat extent: ({lat_min}, {lat_max}), Lon extent: ({lon_min}, {lon_max})")

    # Display the extracted tiepoints
    print("\nTie Points:")
    for tp in tiepoints:
        print(tp)

    return geo_info

# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def relative_transform_im_osm(tie_im_path, tie_osm_path):
    """
    Calculate the relative transformation between two sets of tie points (derived from NovaSAR image and OSM).
    """
    import geopandas as gpd
    import rasterio as rio
    
    # Load the shapefile
    tie_im = gpd.read_file(tie_im_path)
    # Extract coordinates as a list of (x, y) tuples
    tie_im = [(point.x, point.y) for point in tie_im.geometry]
    # tie_im = np.array(tie_im)

    # Load the shapefile
    tie_osm = gpd.read_file(tie_osm_path)
    # Extract coordinates as a list of (x, y) tuples
    tie_osm = [(point.x, point.y) for point in tie_osm.geometry]
    # tie_osm = np.array(tie_osm)

    tiepoints = []
    for tie_osm_ii, tie_im_ii in zip(tie_osm, tie_im):
        lon_osm_ii, lat_osm_ii = tie_osm_ii[0], tie_osm_ii[1]
        lon_im_ii,  lat_im_ii  = tie_im_ii[0],  tie_im_ii[1]

        tiepoints.append( rio.control.GroundControlPoint( row=lat_im_ii, col=lon_im_ii, x=lon_osm_ii, y=lat_osm_ii) )

    # Generate new transform using the new tiepoints information extracted from the metadata XML file:
    delta_transform = rio.transform.from_gcps(tiepoints)
    print("Relative Transform:", delta_transform)

    return delta_transform


# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


def new_geotif(tif_file_path, transform, crs, out_file_path):
    """
    Create a new GeoTIFF file with the specified transformation and CRS.
    
    Parameters:
    - tif_file: Path to the input TIFF file.
    - transform: The transformation matrix to apply.
    - crs: The coordinate reference system to use.
    - out_file: path for the output file.
    """
    import rasterio as rio
    src = rio.open(tif_file_path)
    im = src.read(1)
    # im_uint8 = (255*im).astype(np.uint8)

    # Save the array as a GeoTIFF
    profile = src.profile
    profile.update(
        crs=crs,  
        transform=transform,
        compress='lzw'
    )

    with rio.open(
        out_file_path,
        'w',
        **profile
    ) as dst:
        dst.write(im, 1)

    print(f"\nSaved corrected image to {out_file_path}")