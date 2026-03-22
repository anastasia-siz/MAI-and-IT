import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

# Физические константы
GM = 398600.4415e9  # Гравитационный параметр Земли, м^3/с^2
R_earth = 6371000   # Радиус Земли, м

# Параметры КА
mass = 1000         # масса КА, кг
Cd = 2.2            # коэффициент аэродинамического сопротивления
A = 2.0             # характерная площадь, м^2

# Функция плотности атмосферы
def atmospheric_density(h):
    if h < 0:
        return 0
    return 1e-13 * np.exp(-h / 100000)

# Уравнения движения
def equations_of_motion(state):
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
    
    return np.array([
        vx, vy, vz,
        gravity[0] + drag[0],
        gravity[1] + drag[1],
        gravity[2] + drag[2]
    ])

# Метод Рунге-Кутта 4-го порядка
def runge_kutta_4(state, dt):
    k1 = dt * equations_of_motion(state)
    k2 = dt * equations_of_motion(state + 0.5 * k1)
    k3 = dt * equations_of_motion(state + 0.5 * k2)
    k4 = dt * equations_of_motion(state + k3)
    return state + (k1 + 2*k2 + 2*k3 + k4) / 6

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
dt = 60  # шаг 60 секунд
total_time = 86400 * 2  # 2 дня
num_steps = int(total_time / dt)

# Основная логика программы
def main():
    file_path = 'file.xlsx'  # путь к файлу
    initial_conditions = read_initial_conditions(file_path)
    
    # Создаем список для хранения всех траекторий
    all_trajectories = []
    
    for i, initial_state in enumerate(initial_conditions):
        print(f"\nРасчет траектории {i+1} из {len(initial_conditions)}")
        
        # Инициализация массива для хранения результатов
        trajectory = np.zeros((num_steps + 1, 6))
        trajectory[0] = initial_state
        
        # Интегрирование
        state = initial_state
        for j in range(num_steps):
            state = runge_kutta_4(state, dt)
            trajectory[j + 1] = state
        
        # Сохраняем траекторию
        all_trajectories.append(trajectory)
        
        # Выводим результаты
        print(f"\nНачальные условия для траектории {i+1}:")
        print(f"Положение: ({initial_state[0]:.3f}, {initial_state[1]:.3f}, {initial_state[2]:.3f}) м")
        print(f"Скорость: ({initial_state[3]:.3f}, {initial_state[4]:.3f}, {initial_state[5]:.3f}) м/с")
        
        final_state = trajectory[-1]
        print("\nКонечные условия:")
        print(f"Положение: ({final_state[0]:.3f}, {final_state[1]:.3f}, {final_state[2]:.3f}) м")
        print(f"Скорость: ({final_state[3]:.3f}, {final_state[4]:.3f}, {final_state[5]:.3f}) м/с")
    
    # Визуализация всех траекторий
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i, trajectory in enumerate(all_trajectories):
        ax.plot(trajectory[:,0], trajectory[:,1], trajectory[:,2], label=f'Траектория {i+1}')
    ax.set_xlabel('X (м)')
    ax.set_ylabel('Y (м)')
    ax.set_zlabel('Z (м)')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()

