import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from scipy.integrate import solve_ivp
from astropy.time import Time
from astropy import units as u
from astropy.coordinates import solar_system_ephemeris, get_body_barycentric, ICRS, GCRS
from astropy.coordinates import CartesianRepresentation

# Активируем эфемериды (используем встроенные для надёжности)
try:
    solar_system_ephemeris.set('builtin')
except Exception as e:
    print(f"Ошибка загрузки эфемерид: {e}")
   
# Физические константы (в км и с для удобства)
GM = 398600.4415  # км^3/с^2
R_earth = 6371  # км
J2 = 1.08262668e-3
mu_M = 4902.800076  # км^3/с^2
mu_S = 132712440018  # км^3/с^2

# Параметры КА
mass = 1000  # кг
Cd = 2.2
A = 2.0  # м^2

# Функция плотности атмосферы (в км)
def atmospheric_density(h_km, F107=150, Ap=4):

    if h_km < 0:
        return 0.0

    if h_km <= 120:
        # Тропосфера и стратосфера — упрощённая аппроксимация
        rho = 1.225 * np.exp(-h_km / 8.5)
    elif h_km <= 200:
        # Мезосфера и нижняя термосфера
        h_ref = 120.0
        rho_ref = 8.9e-7  # кг/м³ на высоте 120 км
        scale_height = 55.0 + 0.3 * (F107 - 150) + 2.0 * (Ap - 4)
        rho = rho_ref * np.exp(-(h_km - h_ref) / scale_height)
    elif h_km <= 500:
        # Верхняя термосфера
        h_ref = 200.0
        rho_ref = 3.5e-10  # кг/м³ на высоте 200 км
        scale_height = 65.0 + 0.4 * (F107 - 150) + 3.0 * (Ap - 4)
        rho = rho_ref * np.exp(-(h_km - h_ref) / scale_height)
    else:
        # Экзосфера — экспоненциальное убывание
        h_ref = 500.0
        rho_ref = 1.0e-14  # кг/м³ на высоте 500 км
        scale_height = 80.0 + 0.5 * (F107 - 150)
        rho = rho_ref * np.exp(-(h_km - h_ref) / scale_height)

    return rho

# Функция положения Луны и Солнца с использованием Astropy
def get_moon_sun_positions(t_seconds):
    # Преобразуем секунды в астрономический формат времени
    t_astropy = Time('2023-01-01T00:00:00') + t_seconds * u.second

    try:
        # Получаем позиции из эфемерид
        r_moon = get_body_barycentric('moon', t_astropy).xyz.to(u.km).value
        r_sun = get_body_barycentric('sun', t_astropy).xyz.to(u.km).value
    except Exception as e:
        print(f"Ошибка получения позиций: {e}")
    return r_moon, r_sun

# Уравнения движения 
def equations_of_motion(t, state):
    x, y, z, vx, vy, vz = state
    r = np.array([x, y, z])
    r_mag = np.linalg.norm(r)

    # Гравитационное ускорение
    r_unit = r / r_mag
    gravity = -GM * r_unit / r_mag**2

    # Сопротивление (только на низких высотах)
    h = r_mag - R_earth
    if h < 500:  # Только ниже 500 км учитываем атмосферу
        rho = atmospheric_density(h)
        v = np.array([vx, vy, vz])
        v_mag = np.linalg.norm(v)
        drag = -0.5 * Cd * A * rho * v_mag * v / mass
    else:
        drag = np.zeros(3)

    # Возмущение J2 
    if r_mag > 0:  # Защита от деления на ноль
        J2_term = (3 * GM * R_earth**2 * J2 / (2 * r_mag**3)) * np.array([
            x * (5 * z**2 / r_mag**2 - 1),
            y * (5 * z**2 / r_mag**2 - 1),
            z * (5 * z**2 / r_mag**2 - 3)
        ]) 
    else:
        J2_term = np.zeros(3)

    # Положения Луны и Солнца
    r_moon, r_sun = get_moon_sun_positions(t)

    # Возмущение от Луны
    r_moon_rel = r_moon - r
    accel_moon = mu_M * r_moon_rel / np.linalg.norm(r_moon_rel)**3

    # Возмущение от Солнца
    r_sun_rel = r_sun - r
    accel_sun = mu_S * r_sun_rel / np.linalg.norm(r_sun_rel)**3

    # Полное ускорение
    accel_total = gravity + drag + J2_term + accel_moon + accel_sun

    return np.array([
        vx, vy, vz,
        accel_total[0],
        accel_total[1],
        accel_total[2]
    ])







def get_nutation_matrix(jd):
    """Вычисляет матрицу нутации N для перехода от истинной эпохи к средней"""
    T = (jd - 2451545.0) / 36525.0
    O = np.radians(125.0445 - 1934.1363 * T)
    L = np.radians(280.4665 + 36000.7698 * T)
    d_psi = np.radians(-17.20 * np.sin(O) - 1.32 * np.sin(2*L)) / 3600.0
    d_eps = np.radians(9.20 * np.cos(O) + 0.57 * np.cos(2*L)) / 3600.0
    eps0 = np.radians(23.439291 - 0.0130042 * T)
    eps_true = eps0 + d_eps
    Rx1 = np.array([[1, 0, 0], [0, np.cos(eps0), np.sin(eps0)], [0, -np.sin(eps0), np.cos(eps0)]])
    Rz = np.array([[np.cos(-d_psi), -np.sin(-d_psi), 0], [np.sin(-d_psi), np.cos(-d_psi), 0], [0, 0, 1]])
    Rx2 = np.array([[1, 0, 0], [0, np.cos(-eps_true), np.sin(-eps_true)], [0, -np.sin(-eps_true), np.cos(-eps_true)]])
    # итоговая матрица нутации
    return (Rx2 @ Rz @ Rx1)


def get_precession_matrix(jd):
    """Вычисляет матрицу прецессии P"""
    T = (jd - 2451545.0) / 36525.0
    x_p = (2306.2181 * T + 0.30188 * T**2 + 0.017998 * T**3) / 3600.0
    y_p = (2306.2181 * T + 1.09468 * T**2 + 0.018203 * T**3) / 3600.0
    z_p = (2004.3109 * T - 0.42665 * T**2 - 0.041833 * T**3) / 3600.0
    # перевод в радианы для функций sin/cos
    x_p, y_p, z_p = map(np.radians, [x_p, y_p, z_p])
    # матрицы вращения для формирования P
    Rz1 = np.array([[np.cos(-x_p), -np.sin(-x_p), 0],
                    [np.sin(-x_p),  np.cos(-x_p), 0],
            [0,              0,             1]])
    Ry = np.array([[ np.cos(z_p),  0,  np.sin(z_p)],
                  [ 0,              1,  0],
                  [-np.sin(z_p),  0,  np.cos(z_p)]])
    Rz2 = np.array([[np.cos(-z_p),    -np.sin(-z_p),    0],
            [np.sin(-z_p),     np.cos(-z_p),    0],
            [0,              0,             1]])
    # матрица P переводит из J2000 в текущую эпоху jnow
    P = Rz2 @ Ry @ Rz1
    return P.T

def get_gst(jd):
    """Вычисляет GMST (Greenwich Mean Sidereal Time) — звёздное время в Гринвиче"""
    T = (jd - 2451545.0) / 36525.0
    gst = 280.46061837 + 360.98564736629 * (jd - 2451545.0) + 0.000387933 * T**2
    return np.radians(gst % 360)

def horizontal_to_j2000(az, el, dist, lat, lon, jd):
    """Преобразует горизонтальные координаты в систему J2000"""
    az_r, el_r = np.radians(az), np.radians(el)
    v_loc = np.array([
        dist * np.cos(el_r) * np.cos(az_r),
        dist * np.cos(el_r) * np.sin(az_r),
        dist * np.sin(el_r)
    ])
    # учёт широты
    lat_r = np.radians(lat)
    R_lat = np.array([
        [-np.sin(lat_r), 0, np.cos(lat_r)],
        [0,              1, 0],
        [np.cos(lat_r),  0, np.sin(lat_r)]
    ])
    v_fixed = R_lat @ v_loc
    # учёт долготы и вращения Земли
    lst = get_gst(jd) + np.radians(lon)
    y_lst = np.array([
        [np.cos(lst), -np.sin(lst), 0],
        [np.sin(lst),  np.cos(lst), 0],
        [0,            0,           1]
    ])
    z_nut = y_lst @ v_fixed
    # учёт нутации
    N_inv = get_nutation_matrix(jd)
    v_prec = N_inv @ z_nut
    # учёт прецессии (jnow -> J2000)
    P_inv = get_precession_matrix(jd)
    v_j2000 = P_inv @ v_prec
    return v_j2000

def j2000_to_geo(v_j2000, lat, lon, jd):
    """Преобразует вектор из инерциальной системы J2000 в горизонтальную систему координат (Az, El, Dist)"""
    # 1. Обратная прецессия (J2000 -> Jnow)
    P = get_precession_matrix(jd)
    v_prec = P @ v_j2000
    # 2. Обратная нутация (Средняя эпоха -> Истинная эпоха)
    N = get_nutation_matrix(jd)
    z_nut = N.T @ v_prec
    # 3. Обратный учёт вращения Земли и долготы
    lst = get_gst(jd) + np.radians(lon)
    y_lst = np.array([
        [np.cos(lst), -np.sin(lst), 0],
        [np.sin(lst),  np.cos(lst), 0],
        [0,            0,           1]
    ])
    v_fixed = y_lst.T @ z_nut
    # 4. Обратный учёт широты
    lat_r = np.radians(lat)
    R_lat = np.array([
        [-np.sin(lat_r), 0, np.cos(lat_r)],
        [0,              1, 0],
        [np.cos(lat_r),  0, np.sin(lat_r)]
    ])
    v_loc = R_lat.T @ v_fixed
    # 5. Переход из декартовых координат в сферические
    dist = np.linalg.norm(v_loc)
    el_r = np.arcsin(v_loc[2] / dist)
    az_r = np.arctan2(v_loc[1], v_loc[0])
    return np.degrees(az_r) % 360, np.degrees(el_r), dist


# Функция преобразования из J2000 (ICRS) в ГЦСК (GCRS)
def j2000_to_gcrs(position, velocity, t_seconds):
    """
    Преобразует координаты из J2000 (ICRS) в ГЦСК (GCRS)
    position, velocity — массивы [x, y, z] в км и км/с
    t_seconds — время в секундах от начала отсчёта
    """
    t_astropy = Time('2023-01-01T00:00:00') + t_seconds * u.second
    # Создаём координату в ICRS (J2000)
    icrs_coord = ICRS(
        x=position[0] * u.km,
        y=position[1] * u.km,
        z=position[2] * u.km,
        v_x=velocity[0] * u.km/u.s,
        v_y=velocity[1] * u.km/u.s,
        v_z=velocity[2] * u.km/u.s,
        representation_type=CartesianRepresentation,
        differential_type='cartesian'
    )
    # Преобразуем в GCRS (ГЦСК)
    gcrs_coord = icrs_coord.transform_to(GCRS(obstime=t_astropy))
    # Извлекаем позиции и скорости в ГЦСК
    pos_gcrs = gcrs_coord.cartesian.xyz.to(u.km).value
    vel_gcrs = gcrs_coord.velocity.d_xyz.to(u.km/u.s).value
    return pos_gcrs, vel_gcrs




# Основная логика программы
def main():
    initial_state = np.array([5663505034.000, 743916736.000, -3824664144.000, 4347718.000, -1035138.000, 6256515.000])

    total_time = 86400 * 2  # 2 дня в секундах
    t_span = (0, total_time)
    rtol = 1e-12  # относительная погрешность
    atol = 1e-12  # абсолютная погрешность

    all_trajectories = []

    print(f"\nРасчёт траектории")

    # Интегрирование методом DOP853 (RK853)
    solution = solve_ivp(
        fun=equations_of_motion,
        t_span=t_span,
        y0=initial_state,
        method='DOP853',
        rtol=rtol,
        atol=atol,
        dense_output=True
     )

    if solution.success:
        print("Интегрирование успешно завершено")
    else:
        print(f"Ошибка при интегрировании траектории :", solution.message)
        return

    # Сохраняем результаты
    trajectory = np.column_stack([solution.t, solution.y.T])
    all_trajectories.append(trajectory)
    
    # Выводим результаты
    print(f"\nНачальные условия для траектории :")
    print(f"Положение: ({initial_state[0]:.3f}, {initial_state[1]:.3f}, {initial_state[2]:.3f}) км")
    print(f"Скорость: ({initial_state[3]:.3f}, {initial_state[4]:.3f}, {initial_state[5]:.3f}) км/с")

    final_state = solution.y[:, -1]
    final_time = solution.t[-1]

    print("\nКонечные условия (J200):")
    print(f"Положение: ({final_state[0]:.3f}, {final_state[1]:.3f}, {final_state[2]:.3f}) км")
    print(f"Скорость: ({final_state[3]:.3f}, {final_state[4]:.3f}, {final_state[5]:.3f}) км/с")






     # Преобразование конечных данных из J2000 (ICRS) в ГЦСК (GCRS)
    final_position_j2000 = final_state[:3]
    final_velocity_j2000 = final_state[3:]

    pos_gcrs, vel_gcrs = j2000_to_gcrs(
        final_position_j2000,
        final_velocity_j2000,
        final_time
    )

    # Вывод конечных данных в системе ГЦСК
    print("\nКонечные условия в системе ГЦСК:")
    print(f"Положение: ({pos_gcrs[0]:.3f}, {pos_gcrs[1]:.3f}, {pos_gcrs[2]:.3f}) км")
    print(f"Скорость: ({vel_gcrs[0]:.3f}, {vel_gcrs[1]:.3f}, {vel_gcrs[2]:.3f}) км/с")




        
if __name__ == "__main__":
    main()


