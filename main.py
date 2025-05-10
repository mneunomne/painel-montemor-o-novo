import numpy as np
from PIL import Image, ImageDraw
import datetime
import struct
import re
import csv
import math
import os
import cv2

# ====================== DATA EXTRACTION FROM CSV ======================
def read_data_from_csv(csv_file):
    entries = []
    with open(csv_file, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            # Skip empty rows or header if present
            if not row or not row[0].strip():
                continue
                
            # Assuming CSV format: [id,year,location,lat,lng,explorer,description]
            try:
                entry_id = row[0].strip()
                year = int(row[1])
                location = row[2].strip()
                lat = row[3].strip()
                lng = row[4].strip()
                explorer = row[5].strip()
                
                # Convert latitude/longitude to decimal
                lat_val = float(re.sub('[°NS]', '', lat))
                if 'S' in lat:
                    lat_val = -lat_val
                
                lng_val = float(re.sub('[°EW]', '', lng))
                if 'W' in lng:
                    lng_val = -lng_val
                
                # Generate a timestamp (assuming January 1st of the year)
                timestamp = datetime.datetime(year, 1, 1).timestamp()
                
                # Create filename-friendly location name
                clean_location = re.sub(r'[^\w]', '', location).lower()
                filename = f"{entry_id}_{clean_location}"
                
                entries.append({
                    'id': entry_id,
                    'year': year,
                    'location': location,
                    'lat': lat_val,
                    'lng': lng_val,
                    'explorer': explorer,
                    'timestamp': timestamp,
                    'filename': filename
                })
            except (ValueError, IndexError) as e:
                print(f"Skipping malformed row: {row}. Error: {e}")
    return entries

def create_aruco_marker(marker_id, marker_size=16, border_bits=1):
    """Create an ArUco marker using OpenCV"""
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    
    # Generate the marker
    marker_img = np.zeros((marker_size, marker_size), dtype=np.uint8)
    cv2.aruco.generateImageMarker(aruco_dict, marker_id % 50, marker_size, marker_img, border_bits)
    
    # Convert to binary image (0 or 255)
    marker_img = (marker_img > 0).astype(np.uint8) * 255
    
    return marker_img

def float_to_bytes(f):
    return struct.pack('!d', f)


def create_square_tile(data_entry, size=400):
    """Create a square image for a single data entry with a central marker
    
    Parameters:
    - data_entry: Dictionary containing entry data (id, lat, lng, timestamp)
    - size: Size of the final square image in pixels
    """
    import numpy as np
    import math
    from PIL import Image, ImageDraw
    
    # Get marker ID from the data entry
    marker_id = int(data_entry['id']) if data_entry['id'].isdigit() else abs(hash(data_entry['id']) % 10000)
    
    # Encode the data (lat, lng, timestamp)
    encoded_data = float_to_bytes(data_entry['lat']) + \
                   float_to_bytes(data_entry['lng']) + \
                   float_to_bytes(data_entry['timestamp'])
    
    # Convert to numpy array of bytes (uint8)
    byte_values = np.frombuffer(encoded_data, dtype=np.uint8)
    
    # Split each byte into two 4-bit values (0-15)
    int4_values = []
    for byte in byte_values:
        int4_values.append(byte >> 4)         # First 4 bits
        int4_values.append(byte & 0x0F)       # Last 4 bits
    
    int4_array = np.array(int4_values, dtype=np.uint8)
    
    # Calculate the square side length that fits our data (ensure it's even)
    data_length = len(int4_array)
    side = int(math.ceil(math.sqrt(data_length)))
    if side % 2 == 0:  # Ensure even number of rows/cols
        side += 1
    
    # Pad array to make a perfect square
    padded_length = side * side
    if len(int4_array) < padded_length:
        int4_array = np.pad(int4_array, (0, padded_length - len(int4_array)), 'constant')
    
    # Reshape to square
    data_square = int4_array.reshape(side, side)
    
    # Calculate center for marker placement
    center_x = side // 2
    center_y = side // 2
    
    # Create mask for central 2x2 marker area
    #marker_mask = np.zeros((side, side), dtype=bool)
    #marker_mask[center_y:center_y, center_x:center_x] = True
    
    # Scale values for better visibility (0-15 to 0-255)
    data_square = data_square * 17
    
    # Scale to the desired image size
    data_point_size = size // side
    img = Image.new('L', (size, size), color=255)
    
    # Draw data points as squares
    draw = ImageDraw.Draw(img)
    for y in range(side):
        for x in range(side):
            # Skip the center area where the marker will be placed
            if marker_mask[y, x]:
                continue
                
            px = x * data_point_size
            py = y * data_point_size
            
            # Calculate center of the square
            center_px = px + data_point_size // 2
            center_py = py + data_point_size // 2
            
            # Calculate the size of the rotated square
            rotated_size = data_point_size * 0.7
            
            # Calculate the four corners of the rotated square
            half_diag = rotated_size / math.sqrt(2)
            points = [
                (center_px - half_diag, center_py),
                (center_px, center_py - half_diag),
                (center_px + half_diag, center_py),
                (center_px, center_py + half_diag)
            ]
            
            # Draw the rotated square
            draw.polygon(points, fill=int(data_square[y, x]))
    
    # Create and place the central marker
    central_marker_size = int(data_point_size * 1)
    marker = create_aruco_marker(marker_id, marker_size=central_marker_size)
    marker_img = Image.fromarray(marker, 'L')

    # Convert to RGBA
    rgba_img = Image.new("RGBA", img.size)
    rgba_img.paste(img, (0, 0))
    
    # Convert marker to RGBA with transparency
    rgba_marker = Image.new("RGBA", marker_img.size)
    rgba_marker.paste(marker_img, (0, 0))
    
    # Rotate marker with transparency
    rotated_marker = rgba_marker.rotate(45, expand=True, resample=Image.BICUBIC)
    rotated_marker = rotated_marker.resize((central_marker_size, central_marker_size))
    
    # Place the marker in the center
    marker_pos_x = (size // 2) - (central_marker_size // 2)
    marker_pos_y = (size // 2) - (central_marker_size // 2)
    rgba_img.paste(rotated_marker, (marker_pos_x, marker_pos_y), rotated_marker)
    
    return rgba_img

def main():
    # Use navegacoes.csv as default without prompting
    csv_file = "navegacoes.csv"
    
    # Fixed parameters - can be changed in the code directly
    tile_size = 400  # Size of the output image
    
    try:
        # Read and parse the data from CSV
        entries = read_data_from_csv(csv_file)
        
        if not entries:
            print("No valid entries found in the CSV file.")
            return
        
        # Create tiles directory if it doesn't exist
        os.makedirs("tiles", exist_ok=True)
        
        # Process each entry
        for entry in entries:
            # Create the square tile with new requirements
            img = create_square_tile(entry, size=tile_size)
            
            # Save the image
            output_file = f"tiles/{entry['filename']}.png"
            img.save(output_file)
            print(f"Created tile: {output_file}")
            
        print(f"\nSuccessfully created {len(entries)} tiles in the 'tiles' folder.")
            
    except FileNotFoundError:
        print(f"Error: File '{csv_file}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()