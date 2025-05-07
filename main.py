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
    """Create a square image for a single data entry using 4-bit encoding with central marker
    
    Parameters:
    - data_entry: Dictionary containing entry data
    - size: Size of the final square image in pixels
    """
    # Get marker ID from the data entry
    marker_id = int(data_entry['id']) if data_entry['id'].isdigit() else abs(hash(data_entry['id']) % 10000)
    
    # Calculate the spacing pattern based on marker_id (0-15)
    gradient_spacing = marker_id % 16
    
    # Encode the data (lat, lng, timestamp)
    encoded_data = float_to_bytes(data_entry['lat']) + \
                  float_to_bytes(data_entry['lng']) + \
                  float_to_bytes(data_entry['timestamp'])
    
    # Convert to numpy array of bytes (uint8)
    byte_values = np.frombuffer(encoded_data, dtype=np.uint8)
    
    # Split each byte into two 4-bit values (0-15)
    int4_values = []
    for byte in byte_values:
        # First 4 bits (shift right by 4)
        int4_values.append(byte >> 4)
        # Last 4 bits (mask with 0x0F)
        int4_values.append(byte & 0x0F)
    
    int4_array = np.array(int4_values, dtype=np.uint8)
    
    # Calculate the square that fits our data (ensure it's even)
    data_length = len(int4_array)
    side = int(math.ceil(math.sqrt(data_length)))
    if side % 2 == 1:  # Ensure even number of rows/cols
        side += 1
    
    padded_length = side * side
    
    # Pad with zeros if needed to make a square
    if len(int4_array) < padded_length:
        int4_array = np.pad(int4_array, (0, padded_length - len(int4_array)), 'constant')
    
    # Reshape to square
    data_square = np.zeros((side, side), dtype=np.uint8)
    
    # Insert gradient reference points into the data array
    # We'll place 16 gradient points (0-15) in a pattern determined by the gradient_spacing
    
    # Base positions for gradient markers (in a cross pattern from the center)
    # These are offsets from the center
    gradient_positions = []
    
    # Calculate center indices
    center_x = side // 2
    center_y = side // 2
    
    # Generate gradient positions in a pattern based on marker_id
    # We'll use a spiral pattern starting from the center
    for i in range(16):  # 16 gradient points (0-15)
        # Calculate distance from center based on gradient spacing
        distance = (i * gradient_spacing) % (side // 2 - 1)
        if distance == 0:
            distance = 1  # Ensure we don't place at center (reserved for marker)
            
        # Calculate spiral position
        angle = 2 * math.pi * i / 16
        offset_x = int(distance * math.cos(angle))
        offset_y = int(distance * math.sin(angle))
        
        # Calculate absolute position (ensure it's within bounds and doesn't overlap with center)
        pos_x = center_x + offset_x
        pos_y = center_y + offset_y
        
        # Constrain to valid indices
        pos_x = max(0, min(side - 1, pos_x))
        pos_y = max(0, min(side - 1, pos_y))
        
        gradient_positions.append((pos_x, pos_y, i))
    
    # Fill the data square with our encoded data
    data_index = 0
    
    # Create mask for central 2x2 marker area
    marker_mask = np.zeros((side, side), dtype=bool)
    marker_mask[center_y-1:center_y+1, center_x-1:center_x+1] = True
    
    # Create mask for gradient points
    gradient_mask = np.zeros((side, side), dtype=bool)
    for x, y, _ in gradient_positions:
        gradient_mask[y, x] = True
    
    # First, place gradient reference points
    for x, y, value in gradient_positions:
        data_square[y, x] = value * 17  # Scale 0-15 to 0-255
    
    # Then, fill in the rest with data (skipping marker and gradient positions)
    for y in range(side):
        for x in range(side):
            if not marker_mask[y, x] and not gradient_mask[y, x]:
                if data_index < len(int4_array):
                    data_square[y, x] = int4_array[data_index] * 17  # Scale 0-15 to 0-255
                    data_index += 1
    
    # Scale to the desired image size
    data_point_size = size // side
    img = Image.new('L', (size, size), color=128)
    
    # Draw data points
    draw = ImageDraw.Draw(img)
    for y in range(side):
        for x in range(side):
            px = x * data_point_size
            py = y * data_point_size
            
            # Skip the center area where the marker will be placed
            if marker_mask[y, x]:
                continue
                
            # Draw the data point
            draw.rectangle(
                [(px, py), (px + data_point_size - 1, py + data_point_size - 1)], 
                fill=int(data_square[y, x])
            )
    
    # Create and place the central marker
    central_marker_size = data_point_size * 2  # 2x2 data points
    marker = create_aruco_marker(marker_id, marker_size=central_marker_size)
    marker_img = Image.fromarray(marker, 'L')
    
    # Calculate center position to place the marker
    marker_x = (center_x - 1) * data_point_size  # -1 because marker is 2x2 and centered
    marker_y = (center_y - 1) * data_point_size
    
    # Place the marker
    img.paste(marker_img, (marker_x, marker_y))
    
    return img

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