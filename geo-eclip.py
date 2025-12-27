import math

def geo_to_ecliptic(x, y, z, epsilon_deg=23.26):

    eps = math.radians(epsilon_deg)
    cos_eps = math.cos(eps)
    sin_eps = math.sin(eps)
    
    # Поворот вокруг оси х, поэтому х не меняется
    x_ecl = x
    y_ecl = y * cos_eps + z * sin_eps
    z_ecl = -y * sin_eps + z * cos_eps
    
    return x_ecl, y_ecl, z_ecl

def res():
    
    test = [
        {"x": 1000, "y": 2000, "z": 3000},
    ]
    for point in test:
        x_ecl, y_ecl, z_ecl = geo_to_ecliptic(
            point["x"], point["y"], point["z"]
        )
        
        print(f"  ГСК: ({point['x']}, {point['y']}, {point['z']})")
        print(f"  Эклиптическая: ({x_ecl:.2f}, {y_ecl:.2f}, {z_ecl:.2f})")
if __name__ == "__main__":
    res()
