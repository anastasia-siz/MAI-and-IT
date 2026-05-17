import numpy as np
from scipy.integrate import solve_ivp
from astropy.time import Time
from astropy import units as u
from astropy.coordinates import EarthLocation, GCRS
from astropy.utils import iers
import json
import math

iers.conf.auto_download = False

GM = 398600.4415
R_earth = 6371.0
J2 = 1.08262668e-3

# ============================================================
# КРУГОВАЯ ОРБИТА С НАКЛОНОМ 55°
# ============================================================
orbit_altitude = 400.0
orbit_radius = R_earth + orbit_altitude
v_circular = math.sqrt(GM / orbit_radius)
inclination = math.radians(55)

# Экваториальная орбита
pos_eq = np.array([orbit_radius, 0.0, 0.0])
vel_eq = np.array([0.0, v_circular, 0.0])

# Поворот вокруг X на угол наклонения
rot_x = np.array([
    [1, 0, 0],
    [0, math.cos(inclination), -math.sin(inclination)],
    [0, math.sin(inclination), math.cos(inclination)]
])

pos_rot = rot_x @ pos_eq
vel_rot = rot_x @ vel_eq

initial_state = np.array([pos_rot[0], pos_rot[1], pos_rot[2],
                          vel_rot[0], vel_rot[1], vel_rot[2]])

speed_check = math.sqrt(vel_rot[0]**2 + vel_rot[1]**2 + vel_rot[2]**2)
print(f"Модуль скорости: {speed_check:.4f} км/с (должен быть {v_circular:.4f})")

orbital_period = 2 * math.pi * orbit_radius / v_circular
total_time = 1 * orbital_period  # Только 1 виток для проверки

print(f"Радиус: {orbit_radius} км, скорость: {v_circular:.4f} км/с")
print(f"Период: {orbital_period/60:.1f} мин")

# ============================================================
# УРАВНЕНИЯ ДВИЖЕНИЯ (БЕЗ АТМОСФЕРЫ И J2 ДЛЯ ЧИСТОТЫ)
# ============================================================
def equations_of_motion(t, state):
    x, y, z, vx, vy, vz = state
    r_mag = math.sqrt(x**2 + y**2 + z**2)
   
    if r_mag < 100:
        return np.zeros(6)
   
    # Только гравитация — идеальная круговая орбита
    factor = -GM / r_mag**3
    ax = factor * x
    ay = factor * y
    az = factor * z
   
    return np.array([vx, vy, vz, ax, ay, az])

print("\n=== ИНТЕГРИРОВАНИЕ ===")

solution = solve_ivp(
    equations_of_motion,
    (0, total_time),
    initial_state,
    method='DOP853',
    rtol=1e-13,
    atol=1e-13,
    dense_output=True,
    max_step=10.0  # Маленький шаг
)

print(f"Готово! Шагов: {len(solution.t)}")

# Проверка расстояний
distances = [math.sqrt(solution.y[0,i]**2 + solution.y[1,i]**2 + solution.y[2,i]**2)
             for i in range(len(solution.t))]
print(f"Расстояния: мин={min(distances):.1f}, макс={max(distances):.1f} км")
print(f"Разброс: {max(distances)-min(distances):.1f} км")

if max(distances) > orbit_radius * 1.5:
    print("\n!!! ОРБИТА НЕСТАБИЛЬНА !!!")
    exit()

if max(distances) - min(distances) > 100:
    print("\n!!! СЛИШКОМ БОЛЬШОЙ РАЗБРОС !!!")
    exit()

# Сохраняем 500 точек для плавности
num_points = 500
step = max(1, len(solution.t) // num_points)
indices = list(range(0, len(solution.t), step))[:num_points]

points = []
for i in indices:
    points.append([
        float(solution.y[0, i]),
        float(solution.y[1, i]),
        float(solution.y[2, i])
    ])

print(f"Точек: {len(points)}")

# Проверка первой и последней
first_dist = math.sqrt(points[0][0]**2 + points[0][1]**2 + points[0][2]**2)
last_dist = math.sqrt(points[-1][0]**2 + points[-1][1]**2 + points[-1][2]**2)
print(f"Первая: r={first_dist:.1f}, последняя: r={last_dist:.1f}")

# Координаты Москвы
def geo_to_gcrs(lat, lon, height, jd):
    t = Time(jd, format='jd')
    loc = EarthLocation(lat=lat*u.deg, lon=lon*u.deg, height=height*u.km)
    return loc.get_itrs(obstime=t).transform_to(GCRS(obstime=t)).cartesian.xyz.to(u.km).value

moscow = geo_to_gcrs(55.7558, 37.6173, 0.15, 2459945.0)
print(f"Москва: ({moscow[0]:.1f}, {moscow[1]:.1f}, {moscow[2]:.1f})")

output = {
    "trajectory": points,
    "flight_speed": 300.0,
    "moscow": {"longitude": 37.6, "latitude": 55.75, "earth_radius": 6371.0}
}

output_path = "C:/Users/Taisiya/Documents/Unreal Projects/test1/trajectory_data.json"
with open(output_path, "w") as f:
    json.dump(output, f, indent=2)

print(f"\nСохранено: {output_path}")
print("=== ГОТОВО ===")
