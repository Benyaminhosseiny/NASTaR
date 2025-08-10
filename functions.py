import warnings
warnings.filterwarnings("ignore")
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
    
    # Group files by their parent folder:
    from collections import defaultdict
    from pathlib import Path

    # Group files by their parent folder
    grouped_tif_files = defaultdict(list)

    for file_path in tif_files:
        folder_name = Path(file_path).parts[-3]
        grouped_tif_files[folder_name].append(file_path)

    # Convert defaultdict to a regular dictionary for easier access
    grouped_tif_files = dict(grouped_tif_files)


    return tif_files, grouped_tif_files

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
    # print("Ellipsoid Name:", ellipsoid)

    # Extracting SampledLineSpacing and SampledPixelSpacing from Image_Attributes

    # Find the "Image_Attributes" element in the XML metadata
    im_attr_info_elem = root.find(".//Image_Attributes")

    # Extract SampledLineSpacing
    elem = im_attr_info_elem.find(".//SampledLineSpacing")
    gsd_line = elem.text if elem is not None else None
    geo_info['gsd_line'] = gsd_line
    # print("Sampled Line Spacing:", gsd_line)

    # Extract SampledPixelSpacing
    elem = im_attr_info_elem.find(".//SampledPixelSpacing")
    gsd_pixel = elem.text if elem is not None else None
    geo_info['gsd_pixel'] = gsd_pixel
    # print("Sampled Pixel Spacing:", gsd_pixel)

    elem = geo_info_elem.find(".//SemiMajorAxis")
    SemiMajorAxis_name = elem.text if elem is not None else None
    geo_info['SemiMajorAxis'] = SemiMajorAxis_name

    elem = geo_info_elem.find(".//SemiMinorAxis")
    SemiMinorAxis_name = elem.text if elem is not None else None
    geo_info['SemiMinorAxis'] = SemiMinorAxis_name

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
    # print("\nTie Points:")
    # for tp in tiepoints:
    #     print(tp)

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
    # print("Relative Transform:", delta_transform)

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

    # print(f"\nSaved corrected image to {out_file_path}")

# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def get_geotif_LatLon_extent(tif_file_path):
    import rasterio as rio
    with rio.open(tif_file_path) as src:
        bounds = src.bounds
        lat_min, lon_min = bounds.bottom, bounds.left
        lat_max, lon_max = bounds.top, bounds.right

        # # [Approach 2: Using external Transform file]: Find Latitude and Longitude extent of the image using the new transform
        # src = rio.open(f'{tif_files[tii][:-12]}image_HH_corrected.tif')
        # transform = src.transform
        # lon0, lat0 = transform * (0, 0) 
        # lon1, lat1 = transform * (0, src.shape[0]-1) 
        # lon2, lat2 = transform * (src.shape[1]-1, 0) 
        # lon3, lat3 = transform * (src.shape[1]-1, src.shape[0]-1) 
        # lon_extent, lat_extent = [lon0,lon1,lon2,lon3], [lat0,lat1,lat2,lat3]
        # lat_max = max(lat_extent)
        # lat_min = min(lat_extent)
        # lon_max = max(lon_extent)
        # lon_min = min(lon_extent)
    return lat_min, lon_min, lat_max, lon_max

# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def load_AIS_df(ais_csv_dir, acqdate, lat_min, lon_min, lat_max, lon_max, time_buffer=[1]):
    """
    Load AIS data from a CSV file and filter it based on acquisition date and spatial extent.
    
    Parameters:
    - ais_csv_dir: Path to the AIS CSV file.
    - acqdate: Acquisition date in 'DD/MM/YYYY HH:MM:SS' format.
    - lat_min, lon_min, lat_max, lon_max: Spatial extent for filtering.
    - time_buffer: List of time buffer values in seconds.
    
    Returns:
    - DataFrame containing filtered AIS data.
    """
    import pandas as pd
    from datetime import datetime, timedelta
    
    lat_col       = 'Latitude'
    lon_col       = 'Longitude'
    timestamp_col = '# Timestamp'
    
    chunks = []
    for t_bii in time_buffer:
        start_time = (datetime.strptime(acqdate, '%d/%m/%Y %H:%M:%S') - timedelta(seconds=t_bii)).strftime('%d/%m/%Y %H:%M:%S')
        end_time   = (datetime.strptime(acqdate, '%d/%m/%Y %H:%M:%S') + timedelta(seconds=t_bii)).strftime('%d/%m/%Y %H:%M:%S')

        for chunk in pd.read_csv(ais_csv_dir, chunksize=500000):
            mask = (chunk[timestamp_col] >= start_time) & (chunk[timestamp_col] <= end_time) & \
                   (chunk[lat_col] >= lat_min) & (chunk[lat_col] <= lat_max) & \
                   (chunk[lon_col] >= lon_min) & (chunk[lon_col] <= lon_max)
            filtered = chunk[mask]
            if not filtered.empty:
                chunks.append(filtered)

    if len(chunks)>0:
           AIS_df = pd.concat(chunks, ignore_index=True)  # concatenate all chunks into a single DataFrame
           AIS_df = AIS_df.drop_duplicates(subset=["Name", "Ship type", "Width", "Length"])  # Remove duplicate entries based on Name, Ship type, Width, and Length
    else:
        print("No AIS data found within the specified time and spatial bounds.")
        AIS_df = []
    
    return AIS_df

# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def AIS_row_col_from_lat_lon(lat_AIS, lon_AIS, im_path):
    """
    Convert latitude and longitude coordinates in AIS points to row and column indices in the satellite image.
    Args:
        lat_AIS (list): List of latitude coordinates.
        lon_AIS (list): List of longitude coordinates.
        im_path (str): Path to the satellite image file.
    Returns:
        tuple: Two lists containing row and column indices corresponding to the latitude and longitude coordinates.
    """
    import rasterio as rio
    src = rio.open(im_path)
    
    row_AIS = []
    col_AIS = []
    for latii, lonii in zip(lat_AIS, lon_AIS):
        # Convert latitude and longitude to row and column indices
        rowii, colii = src.index(lonii, latii)

        
        # Approach2: Use the transformation matrix to convert lat/lon to row/col [When the image transform is not available in the src object]
        # rowii, colii = rio.transform.rowcol(transform, lonii, latii) # or rowii, colii = ~transform * (lonii, latii)

        # print(f"Row: {rowii}, Col: {colii}, Lat: {latii}, Lon: {lonii}")
        
        row_AIS.append(rowii)
        col_AIS.append(colii)

    return row_AIS, col_AIS

# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def ship_patches(im_path, im_name,patch_output_dir, AIS_df, row_AIS, col_AIS, h=64, w=64,uint8=False, plt_ptch=False):
    """
    Extract patches from the image based on AIS data and save them to the specified directory.
    
    Parameters:
    - im_path: Path to the input image file.
    - patch_output_dir: Directory where the patches will be saved.
    - row: List of row indices for AIS points.
    - col: List of column indices for AIS points.
    - h: Half height of the patch.
    - w: Half width of the patch.
    """
    import os
    import rasterio as rio
    from rasterio.windows import Window
    import matplotlib.pyplot as plt
    import shutil
    
    if not os.path.exists(patch_output_dir):
        os.makedirs(patch_output_dir)
    else:
        # If the directory already exists, remove its files and create a new one
        shutil.rmtree(patch_output_dir)
        os.makedirs(patch_output_dir)

    if uint8:
        if not os.path.exists(f"{patch_output_dir}_uint8"):
            os.makedirs(f"{patch_output_dir}_uint8")
        else:
            # If the directory already exists, remove its files and create a new one
            shutil.rmtree(f"{patch_output_dir}_uint8")
            os.makedirs(f"{patch_output_dir}_uint8")

    src = rio.open(im_path)
    im  = src.read(1)

    patch_name_all = []
    ii = 1
    for r_ii, c_ii, sh_t_ii in zip(row_AIS, col_AIS, AIS_df['Ship type']):
        if r_ii-h < 0 or r_ii+h >= im.shape[0] or c_ii-w < 0 or c_ii+w >= im.shape[1]:
            print(f"Skipping patch for row {r_ii}, col {c_ii} due to out of bounds.")
            patch_name_all.append('')
            continue
        else:
            subii = im[r_ii-h:r_ii+h, c_ii-w:c_ii+w]
            # Adjust metadata:
            out_meta = src.meta.copy()
            out_meta.update({
                "height"   : h*2,
                "width"    : w*2,
                "transform": src.window_transform( Window(c_ii-w, r_ii-h, 2*w, 2*h) ),
                "compress" :'lzw'
            })
            # Write the output:
            patch_nameii = f"{im_name}_patch_{ii}_{sh_t_ii}"
            patch_name_all.append(patch_nameii)
            out_nameii   = f"{patch_output_dir}/{patch_nameii}.tif"
            with rio.open(out_nameii, "w", **out_meta) as dest:
                dest.write(subii,1)
            
                subii = subii.astype('float32')
                subii -= subii.mean()
                subii /= 5*subii.std()
                subii += 0.5
                subii = (255 * subii).clip(0, 255).astype('uint8')
                
                out_nameii   = f"{patch_output_dir}_uint8/{patch_nameii}_uint8.tif"
                out_meta.update({
                    "dtype": 'uint8',
                })
                with rio.open(out_nameii, "w", **out_meta) as dest:
                    dest.write(subii,1)

            if plt_ptch:
                plt.figure()
                plt.imshow(subii, cmap='gray', vmin=0, vmax=1500)
                plt.text(h, w, f"{sh_t_ii}", fontsize=10, color='blue', ha='center', va='center')

            ii += 1
    
    # Remove empty directories if they are empty after processing (Because of patches being out of bounds)
    if os.path.isdir(patch_output_dir) and not os.listdir(patch_output_dir):
        os.rmdir(patch_output_dir) 
        os.rmdir(f"{patch_output_dir}_uint8")  

    AIS_df['Patch_name'] = patch_name_all
    
    return AIS_df




# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-  

def ExtractPatchAndAIS(tii_path, AIS_path, h=64, w=64):
    """
    Extract patches and AIS data from NovaSAR images.
    
    Parameters:
    - tii_path: Path to the tif file to process.
    - AIS_path: Path to the AIS data CSV file.
    - h: Half height for patch extraction (default is 64).
    - w: Half width for patch extraction (default is 64).
    
    Returns:
    - None
    """
    # Import dependencies
    import os
    import numpy as np
    import math
    from datetime import datetime, date
    import matplotlib.pyplot as plt
    import rasterio as rio
    from pathlib import Path
    import json
    from rasterio.crs import CRS


    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

    parts = tii_path.parts

    im_name = parts[-1]
    print(f"Sample image (tii) name: {im_name}")
    # print(f"Sample image (tii) path:\n{tii_path}")
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    # Find the data acquisition date from the metadata XML file:

    acqdate0 = f"20{'_'.join(parts[-1].split('_')[-4:-2])}" # or f"20{tif_files[tii][-31:-18]}" # YYYYMMDD_HHMMSS
    dt = datetime.strptime(acqdate0, '%Y%m%d_%H%M%S')

    acqdate = dt.strftime('%d/%m/%Y %H:%M:%S') # DD/MM/YYYY HH:MM:SS
    # print("Raw Data Start Time:", acqdate)

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    # Extract Geo and Image information from the metadata XML file:
    metadata_path = f"{tii_path}/metadata.xml"
    geo_info = geo_info_from_metadata(metadata_path)

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    # Generate CRS and transform using ellipsoid and tiepoints information extracted from the metadata XML file:


    # Use ellipsoid parameters for CRS, fallback to WGS84 if not available
    if geo_info['ellipsoid'] == "WGS84":
        crs = CRS.from_epsg(4326)
    else:
        crs = CRS.from_string(f"+proj=longlat +a={geo_info['SemiMajorAxis_name']} +b={geo_info['SemiMinorAxis_name']} +no_defs")

    transform0 = rio.transform.from_gcps(geo_info['tiepoints'])
    # print("Initial Georeferencing Transformation Matrix:", transform0)

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    # Load the modified tie points (based on OSM and NovaSAR images) from shapefiles

    # Find the index of 'NovaSAR' in the path parts
    parts = tii_path.parts
    # nova_index = parts.index('NovaSAR')

    # nova_path = Path(*parts[:nova_index + 1])
    nova_path = Path(*parts[:-1])

    tie_im_path = f"{nova_path}/modified tie points/im_points.shp"
    tie_osm_path = f"{nova_path}/modified tie points/osm_points.shp"

    delta = relative_transform_im_osm(tie_im_path, tie_osm_path)

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    # Save the delta transform to a JSON file

    # Convert to list and save
    with open(f"{nova_path}/delta_transform.json", "w") as f:
        json.dump(list(delta), f)

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    # Load the delta transform from the saved JSON file
    with open(f"{nova_path}/delta_transform.json", "r") as f:
        delta_values = json.load(f)

    # Reconstruct the Affine transform
    from rasterio.transform import Affine
    delta = Affine(*delta_values)

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    # Corrected transformation matrix
    transform=delta*transform0

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    # Create a new geotiff with the corrected transformation matrix
    new_geotif(tif_file_path = f"{tii_path}/image_HH.tif", transform = transform, crs = crs, out_file_path = f"{tii_path}/image_HH_corrected.tif")

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

    # AIS data Path:

    csv_dir = AIS_path

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    # LatLon extent of the geotiff image
    lat_min, lon_min, lat_max, lon_max = get_geotif_LatLon_extent(f'{tii_path}/image_HH_corrected.tif')

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    time_buffer = [1,2]  # seconds
    AIS_df = load_AIS_df(csv_dir, acqdate, lat_min, lon_min, lat_max, lon_max, time_buffer)
    if len(AIS_df) == 0:
        print("No AIS data found within the specified time and spatial bounds.")
    else:
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
        # Save the filtered AIS data to a CSV file:
        AIS_df.to_csv(f"{tii_path}/AIS.csv", index=False)

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

        # Extract ship patches from the image and save them to a directory
        row_AIS, col_AIS = AIS_row_col_from_lat_lon(lat_AIS = AIS_df['Latitude'], lon_AIS = AIS_df['Longitude'], im_path=f'{tii_path}/image_HH_corrected.tif')

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
        patch_output_dir = f"{tii_path}/ship_patches"
        AIS_df = ship_patches(im_path=f'{tii_path}/image_HH_corrected.tif', 
                            im_name=im_name,
                            patch_output_dir=patch_output_dir, 
                            AIS_df=AIS_df, 
                            row_AIS=row_AIS, 
                            col_AIS=col_AIS, 
                            h=h, 
                            w=w,
                            uint8=True,
                            plt_ptch=False)

        # Save the updated AIS data with patch names:
        AIS_df.to_csv(f"{tii_path}/AIS.csv", index=False)
        print(f"Saved the updated AIS data with patch_names column to {tii_path}/AIS.csv")

