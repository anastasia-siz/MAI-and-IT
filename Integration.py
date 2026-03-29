import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import solve_ivp

# Физические константы
GM = 398600.4415e9  # Гравитационный параметр Земли, м^3/с^2
R_earth = 6371000   # Радиус Земли, м
J2=1.08262668e-3
mu_M = 4902.800076e9  # Гравитационный параметр Луны, м^3/с^2
mu_S = 132712440018e9  # Гравитационный параметр Солнца, м^3/с^2

# Параметры КА
mass = 1000         # масса КА, кг
Cd = 2.2            # коэффициент аэродинамического сопротивления
A = 2.0             # характерная площадь, м^2

# Функция плотности атмосферы
def atmospheric_density(h):
    if h < 0:
        return 0
    return (1.225e-12) * np.exp(-(h-100000) / 65000) #6500 масштабная высота

# Функция положения Луны и Солнца (упрощенно)
def get_moon_sun_positions(t):
    # Здесь должна быть реальная модель движения
     # Для примера используем постоянные векторы
    r_moon = np.array([384400000, 0, 0])  # Упрощенное положение Луны
    r_sun = np.array([149.6e9, 0, 0])  # Упрощенное положение Солнца
    return r_moon, r_sun

# Уравнения движения
def equations_of_motion(t, state):
    x, y, z, vx, vy, vz = state
    r = np.array([x, y, z])
    r_mag = np.linalg.norm(r)
    
    # Гравитационное ускорение
    r_unit = r / r_mag
    gravity = -GM * r_unit / r_mag**2
    
    # Сопротивление
    rho = atmospheric_density(r_mag - R_earth)
    v = np.array([vx, vy, vz])
    v_mag = np.linalg.norm(v)
    drag = -0.5 * Cd * A * rho * v_mag * v / mass
    
    # Возмущение J2
    J2_term = (GM * R_earth**2 * J2 / (2 * r_mag**7)) * ((3*z**2 - r_mag**2/3) * r  - np.array([0, 0, 2*z**2]))


   # Положения Луны и Солнца
    r_moon, r_sun =get_moon_sun_positions(t)
    
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




# Функция для чтения данных из Excel
def read_initial_conditions(file_path, dtype={'x': float, 'y': float, 'z': float,  'Vx': float, 'Vy': float, 'Vz': float}):
    df = pd.read_excel(file_path)
    # Предполагаем, что в файле есть столбцы: x, y, z, Vx, Vy, Vz
    initial_conditions = []
    for index, row in df.iterrows():
        initial_conditions.append(np.array([
            row['x'], row['y'], row['z'],
            row['Vx'], row['Vy'], row['Vz']
        ]))
    return initial_conditions

# Параметры интегрирования
total_time = 86400 * 2  # 2 дня
t_span = (0, total_time)

# Настройки точности для DOP853
rtol = 1e-12  # относительная погрешность
atol = 1e-12  # абсолютная погрешность

# Основная логика программы
def main():
    file_path = 'file.xlsx'  # путь к файлу
    initial_conditions = read_initial_conditions(file_path)
    
    # Создаем список для хранения всех траекторий
    all_trajectories = []
    
    for i, initial_state in enumerate(initial_conditions):
        print(f"\nРасчет траектории {i+1} из {len(initial_conditions)}")
        
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
            print(f"Ошибка при интегрировании траектории {i+1}:", solution.message)
            continue

        # Сохраняем результаты
        trajectory = np.column_stack([solution.t, solution.y.T])
        all_trajectories.append(trajectory)
        
        # Выводим результаты
        print(f"\nНачальные условия для траектории {i+1}:")
        print(f"Положение: ({initial_state[0]:.3f}, {initial_state[1]:.3f}, {initial_state[2]:.3f}) м")
        print(f"Скорость: ({initial_state[3]:.3f}, {initial_state[4]:.3f}, {initial_state[5]:.3f}) м/с")
        
        final_state = trajectory[-1]
        print("\nКонечные условия:")
        print(f"Положение: ({final_state[0]:.3f}, {final_state[1]:.3f}, {final_state[2]:.3f}) м")
        print(f"Скорость: ({final_state[3]:.3f}, {final_state[4]:.3f}, {final_state[5]:.3f}) м/с")
    

if __name__ == "__main__":
    main()

