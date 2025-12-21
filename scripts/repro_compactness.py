from shapely.geometry import Polygon
import math

def _compactness(p: Polygon) -> float:
    if p.length <= 0.0:
        return 0.0
    return (4.0 * math.pi * p.area) / (p.length * p.length)

# 100x1 rectangle
p1 = Polygon([(0,0), (100,0), (100,1), (0,1)])
print(f"100x1 Compactness: {_compactness(p1)}")

# 200x1 rectangle
p2 = Polygon([(0,0), (200,0), (200,1), (0,1)])
print(f"200x1 Compactness: {_compactness(p2)}")

# Threshold is 0.03
print(f"Threshold: 0.03")
