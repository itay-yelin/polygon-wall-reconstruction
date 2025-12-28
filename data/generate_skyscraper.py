import json
import random

def generate_office():
    entities = []
    
    # Configuration
    rows, cols = 10, 10
    cubicle_size = 4.0
    corridor_width = 2.0
    wall_thick = 0.1
    
    current_y = 0.0
    
    for r in range(rows):
        current_x = 0.0
        for c in range(cols):
            # Define Cubicle Box
            x1, y1 = current_x, current_y
            x2, y2 = current_x + cubicle_size, current_y + cubicle_size
            
            # Inner coordinates (Wall thickness)
            xi1, yi1 = x1 + wall_thick, y1 + wall_thick
            xi2, yi2 = x2 - wall_thick, y2 - wall_thick
            
            # Introduce slight "CAD Drift" (random epsilon)
            drift = lambda: random.uniform(-0.001, 0.001)
            
            # Outer Shell
            entities.extend([
                {"type": "line", "start_point": {"x": x1, "y": y1}, "end_point": {"x": x2+drift(), "y": y1}},
                {"type": "line", "start_point": {"x": x2, "y": y1}, "end_point": {"x": x2, "y": y2+drift()}},
                {"type": "line", "start_point": {"x": x2, "y": y2}, "end_point": {"x": x1+drift(), "y": y2}},
                {"type": "line", "start_point": {"x": x1, "y": y2}, "end_point": {"x": x1, "y": y1+drift()}},
            ])
            
            # Inner Shell (The Cubicle Walls)
            # Leave a gap for the door on the bottom wall!
            # Door from x_start+1.0 to x_start+2.0
            door_start = xi1 + 1.0
            door_end = xi1 + 2.0
            
            entities.extend([
                # Bottom wall (Split by door)
                {"type": "line", "start_point": {"x": xi1, "y": yi1}, "end_point": {"x": door_start, "y": yi1}},
                {"type": "line", "start_point": {"x": door_end, "y": yi1}, "end_point": {"x": xi2, "y": yi1}},
                
                # Right
                {"type": "line", "start_point": {"x": xi2, "y": yi1}, "end_point": {"x": xi2, "y": yi2}},
                # Top
                {"type": "line", "start_point": {"x": xi2, "y": yi2}, "end_point": {"x": xi1, "y": yi2}},
                # Left
                {"type": "line", "start_point": {"x": xi1, "y": yi2}, "end_point": {"x": xi1, "y": yi1}},
            ])
            
            current_x += cubicle_size + wall_thick
            
            # Add vertical corridor every 5 cubicles
            if (c + 1) % 5 == 0:
                current_x += corridor_width

        current_y += cubicle_size + wall_thick
        # Add horizontal corridor every 5 rows
        if (r + 1) % 5 == 0:
            current_y += corridor_width

    output = {
        "name": "Generated Huge Office Floor",
        "metadata": {"generated": True, "line_count": len(entities)},
        "entities": entities
    }
    
    import os
    output_path = os.path.join("data", "huge_office_floor.json")
    
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
        
    print(f"Generated {output_path} with {len(entities)} lines.")

if __name__ == "__main__":
    generate_office()