import math

def spherical_to_geocentric(azimuth, elevation, distance):
#Угол места (elevation, элевация - угловая высота наблюдаемого объекта ): отсчитывается от горизонта (0°) до зенита (90°)

#Зенитный угол (zenit): отсчитывается от зенита (0°) до горизонта (90°)
    # Переводим градусы в радианы
    az_rad = math.radians(azimuth)
    el_rad = math.radians(elevation)
    
    # Зенитный угол = 90° - угол места
    zenit = math.pi/2 - el_rad
    
    # Вычисляем координаты
    X = distance * math.sin(zenit) * math.cos(az_rad)
    Y = distance * math.sin(zenit) * math.sin(az_rad)
    Z = distance * math.cos(zenit)
    
    return X, Y, Z

if __name__ == "__main__": # использование

    az = 45.0    
    el = 30.0    
    r = 1000.0   

    X, Y, Z = spherical_to_geocentric(az, el, r)
    print(f"  Сфер: az={az}°, el={el}°, r={r} км")
    print(f"  ГСК: X={X:.2f} км, Y={Y:.2f} км, Z={Z:.2f} км")
