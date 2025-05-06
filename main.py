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

def add_margin_with_features(image, tile_id, margin_size=32, marker_size=16):
    """Add margin with ArUco markers and grayscale reference"""
    width, height = image.size
    new_width = width + margin_size * 2
    new_height = height + margin_size * 2
    
    # Create new image with margin
    new_image = Image.new('L', (new_width, new_height), color=128)
    new_image.paste(image, (margin_size, margin_size))
    
    # Create drawing context
    draw = ImageDraw.Draw(new_image)
    
    # Calculate positions for markers to ensure they fit within margins
    marker_padding = (margin_size - marker_size) // 2
    
    # Ensure marker fits in margin
    if marker_size > margin_size:
        marker_size = margin_size
        marker_padding = 0
        print(f"Warning: Marker size adjusted to {marker_size} to fit within margins")
    
    # Calculate marker positions
    tl_pos = (marker_padding, marker_padding)
    tr_pos = (new_width - marker_padding - marker_size, marker_padding)
    bl_pos = (marker_padding, new_height - marker_padding - marker_size)
    br_pos = (new_width - marker_padding - marker_size, new_height - marker_padding - marker_size)
    
    # Generate ArUco markers for each corner (using different IDs based on tile_id)
    marker_id_base = int(tile_id) if tile_id.isdigit() else abs(hash(tile_id) % 10000)
    
    # Add white margin around ArUco markers
    white_margin = 2  # 2-pixel white margin
    
    # Top-left marker with white border
    print(f"Creating marker with ID: {marker_id_base}")
    marker_tl = create_aruco_marker(marker_id_base, marker_size - 2*white_margin)
    # Create white-bordered marker
    marker_with_border = np.ones((marker_size, marker_size), dtype=np.uint8) * 255
    # Place marker in center of white border
    h, w = marker_tl.shape
    marker_with_border[white_margin:white_margin+h, white_margin:white_margin+w] = marker_tl
    new_image.paste(Image.fromarray(marker_with_border, 'L'), tl_pos)
    
    # Top-right marker with white border
    marker_tr = create_aruco_marker(marker_id_base + 1, marker_size - 2*white_margin)
    marker_with_border = np.ones((marker_size, marker_size), dtype=np.uint8) * 255
    h, w = marker_tr.shape
    marker_with_border[white_margin:white_margin+h, white_margin:white_margin+w] = marker_tr
    new_image.paste(Image.fromarray(marker_with_border, 'L'), tr_pos)
    
    # Bottom-left marker with white border
    marker_bl = create_aruco_marker(marker_id_base + 2, marker_size - 2*white_margin)
    marker_with_border = np.ones((marker_size, marker_size), dtype=np.uint8) * 255
    h, w = marker_bl.shape
    marker_with_border[white_margin:white_margin+h, white_margin:white_margin+w] = marker_bl
    new_image.paste(Image.fromarray(marker_with_border, 'L'), bl_pos)
    
    # Bottom-right marker with white border
    marker_br = create_aruco_marker(marker_id_base + 3, marker_size - 2*white_margin)
    marker_with_border = np.ones((marker_size, marker_size), dtype=np.uint8) * 255
    h, w = marker_br.shape
    marker_with_border[white_margin:white_margin+h, white_margin:white_margin+w] = marker_br
    new_image.paste(Image.fromarray(marker_with_border, 'L'), br_pos)
    
    # Calculate gradient bar positions - make them same size as markers
    grad_bar_width = marker_size
    
    # Left and right margins - vertical gradient bars
    left_grad_x = (margin_size - marker_size) // 2
    right_grad_x = new_width - margin_size + (margin_size - marker_size) // 2
    
    # Position vertical gradient bars in the middle of left/right margins
    left_grad_y_start = margin_size
    left_grad_y_end = new_height - margin_size
    
    # Draw vertical gradient bars in 16 steps to correspond to 4-bit values (0-15)
    step_count = 16
    step_height = (left_grad_y_end - left_grad_y_start) // step_count
    
    for step in range(step_count):
        y_start = left_grad_y_start + step * step_height
        y_end = left_grad_y_start + (step + 1) * step_height if step < step_count - 1 else left_grad_y_end
        
        # Calculate value for this step (0-255 in 16 steps)
        val = int(255 * step / (step_count - 1))
        
        # Fill the entire step height with the same value
        for y in range(y_start, y_end):
            draw.rectangle([(left_grad_x, y), (left_grad_x + grad_bar_width - 1, y)], fill=val)  # Left reference
            draw.rectangle([(right_grad_x, y), (right_grad_x + grad_bar_width - 1, y)], fill=255-val)  # Right reference
    
    # Top and bottom margins - horizontal gradient bars
    top_grad_y = (margin_size - marker_size) // 2
    bottom_grad_y = new_height - margin_size + (margin_size - marker_size) // 2
    
    # Position horizontal gradient bars in the middle of top/bottom margins
    top_grad_x_start = margin_size
    top_grad_x_end = new_width - margin_size
    
    # Draw horizontal gradient bars in 16 steps
    step_count = 16
    step_width = (top_grad_x_end - top_grad_x_start) // step_count
    
    for step in range(step_count):
        x_start = top_grad_x_start + step * step_width
        x_end = top_grad_x_start + (step + 1) * step_width if step < step_count - 1 else top_grad_x_end
        
        # Calculate value for this step (0-255 in 16 steps)
        val = int(255 * step / (step_count - 1))
        
        # Fill the entire step width with the same value
        for x in range(x_start, x_end):
            draw.rectangle([(x, top_grad_y), (x, top_grad_y + grad_bar_width - 1)], fill=val)  # Top reference
            draw.rectangle([(x, bottom_grad_y), (x, bottom_grad_y + grad_bar_width - 1)], fill=255-val)  # Bottom reference
    
    return new_image

def create_aruco_marker(marker_id, marker_size=16, border_bits=1):
    """Create an ArUco marker using OpenCV"""
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    
    # Generate the marker
    marker_img = np.zeros((marker_size, marker_size), dtype=np.uint8)
    cv2.aruco.generateImageMarker(aruco_dict, marker_id % 50, marker_size, marker_img, border_bits)
    
    # Convert to binary image (0 or 255)
    marker_img = (marker_img > 0).astype(np.uint8) * 255
    
    return marker_img

def create_square_tile(data_entry, size=100, margin_size=32, marker_size=16):
    """Create a square image for a single data entry using 4-bit encoding
    
    Parameters:
    - data_entry: Dictionary containing entry data
    - size: Size of the data square in pixels
    - margin_size: Size of the margin in pixels
    - marker_size: Size of the ArUco markers in pixels
    """
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
    
    # Calculate the largest square that fits our data
    data_length = len(int4_array)
    original_side = int(math.ceil(math.sqrt(data_length)))
    padded_length = original_side * original_side
    
    # Pad with zeros if needed to make a square
    if len(int4_array) < padded_length:
        int4_array = np.pad(int4_array, (0, padded_length - len(int4_array)), 'constant')
    
    # Reshape to original square
    original_square = int4_array[:padded_length].reshape((original_side, original_side))
    
    # Scale values from 0-15 to 0-255 by multiplying by 17 (255/15)
    original_square = original_square * 17
    
    # Scale up to desired size using nearest neighbor interpolation
    img = Image.fromarray(original_square, mode='L')
    img = img.resize((size, size), Image.NEAREST)
    
    # Add margin with features
    img_with_margin = add_margin_with_features(img, data_entry['id'], margin_size, marker_size)
    
    return img_with_margin

def float_to_bytes(f):
    return struct.pack('!d', f)

def main():
    # Use navegacoes.csv as default without prompting
    csv_file = "navegacoes.csv"
    
    # Fixed parameters - can be changed in the code directly
    margin_size = 64
    marker_size = 64
    tile_size = 200
    
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
            # Create the square tile with custom parameters
            img = create_square_tile(entry, size=tile_size, 
                                   margin_size=margin_size, 
                                   marker_size=marker_size)
            
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