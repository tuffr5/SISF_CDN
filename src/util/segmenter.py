import numpy as np
import sys
import argparse
from cloudvolume import CloudVolume
from cloudvolume.exceptions import (
    UnsupportedFormatError,
    InfoUnavailableError,
    UnsupportedProtocolError,
    ScaleUnavailableError,
    OutOfBoundsError,
    SubvoxelVolumeError
)
import json
from requests.exceptions import ConnectionError
from tifffile import imwrite
import tifffile

def main():   
    parser = argparse.ArgumentParser(
                        prog='CloudVolume Segmenter',
                        description='Save region of interest as tif file')

    # Positional arguments
    parser.add_argument('url', help='cloudvolume source')
    parser.add_argument('resolution', type=int, help='an existing MIP')
    parser.add_argument('x1', type=int, help='beginning of x range')
    parser.add_argument('x2', type=int, help ='end of x range')
    parser.add_argument('y1', type=int, help ='beginning of y range')
    parser.add_argument('y2', type=int, help ='end of y range')
    parser.add_argument('z1', type=int, help ='beginning of z range')
    parser.add_argument('z2', type=int, help ='end of z range')
    
    args = parser.parse_args()
    print(f"Parsed arguments: {args}")

    try:
        vol = CloudVolume(args.url, mip = args.resolution, parallel=True, progress=True)
    except UnsupportedFormatError:
        print("Unsupported format error: invalid URL")
        sys.exit(1)
    except InfoUnavailableError:
        print("Info unavailable error: invalid URL")
        sys.exit(1)
    except UnsupportedProtocolError:
        print("Unsupported protocol error: invalid URL")
        sys.exit(1)
    except json.JSONDecodeError:
        print("JSON decode error: invalid URL")
        sys.exit(1)
    except ConnectionError:
        print("Connection error: invalid URL")
        sys.exit(1)
    except Exception as e: # If incorrect MIP, will print MIP x not found and prints available MIPs
        print(f"{type(e).__name__}: {e}")
        sys.exit(1)

    xMax, yMax, zMax = vol.scales[args.resolution]['size']
    
    try:
        if vol.shape[3] == 1: # Grayscale images
            roi = vol[args.x1:args.x2, args.y1:args.y2, args.z1:args.z2]
            imwrite('segmented.tif', roi, photometric='minisblack')
        else: # RGB images
            roi = vol[args.x1:args.x2, args.y1:args.y2, args.z1:args.z2, :3]
            with tifffile.TiffWriter('segmented.tif') as tiff_writer: 
                tiff_writer.write(roi)               
    except OutOfBoundsError:
        print(f"Out of bounds error: requested ranges exceed image bounds.\nImage bounds: {xMax}, {yMax}, {zMax}")
        sys.exit(1)
    except SubvoxelVolumeError:
        print(f"Subvoxel volume error: Requested ranges result in a bounding box with less than one voxel of volume.\nImage bounds: {xMax}, {yMax}, {zMax}")
        sys.exit(1)
    except Exception as e:
        print(f"{type(e).__name__}: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
