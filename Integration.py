import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from scipy.integrate import solve_ivp
from astropy.time import Time
from astropy import units as u
from astropy.coordinates import solar_system_ephemeris, get_body_barycentric

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

# Основная логика программы
def main():
    initial_state = np.array([5663505034.000, 743916736.000, -3824664144.000, 4347718.000, -1035138.000, 6256515.000]) # ЗДЕСЬ МОЖНО МЕНЯТЬ ДАННЫЕ x, y, z, Vx, Vy, Vz

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
    print("\nКонечные условия:")
    print(f"Положение: ({final_state[0]:.3f}, {final_state[1]:.3f}, {final_state[2]:.3f}) км")
    print(f"Скорость: ({final_state[3]:.3f}, {final_state[4]:.3f}, {final_state[5]:.3f}) км/с")
        
if __name__ == "__main__":
    main()


