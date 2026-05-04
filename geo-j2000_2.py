import numpy as np



def get_nutation_matrix(jd):
    """
    вычисляет матрицу нутации N для перехода от истинной эпоxи к средней 
    """
    T = (jd - 2451545.0) / 36525.0
    
    #основные углы
    #долгота узла луны (O) и долгота солнца (L) 

    O = np.radians(125.0445 - 1934.1363 * T)
    L = np.radians(280.4665 + 36000.7698 * T)
    
    #нутация по долготе (d_psi) и наклону (d_eps) в радианах
    d_psi = np.radians((-17.20 * np.sin(O) - 1.32 * np.sin(2*L)) / 3600.0)
    d_eps = np.radians((9.20 * np.cos(O) + 0.57 * np.cos(2*L)) / 3600.0)
    
    #средний наклон экватора к эклиптике
    eps0 = np.radians(23.439291 - 0.0130042 * T)
    eps_true = eps0 + d_eps
    
    #матрицы вращения для формирования N
    Rx1 = np.array([[1, 0, 0], [0, np.cos(eps0), np.sin(eps0)], [0, -np.sin(eps0), np.cos(eps0)]])
    Rz  = np.array([[np.cos(-d_psi), -np.sin(-d_psi), 0], [np.sin(-d_psi), np.cos(-d_psi), 0], [0, 0, 1]])
    Rx2 = np.array([[1, 0, 0], [0, np.cos(-eps_true), np.sin(-eps_true)], [0, -np.sin(-eps_true), np.cos(-eps_true)]])
    
    #итоговая матрица нутации
    return (Rx2 @ Rz @ Rx1)




def get_precession_matrix(jd): #функция вычисляет матрицу прецессии P
    """
    
    прецессия это медленное скольжение земной оси по конусу
    из-за этого полюс мира и точка весеннего равноденствия постоянно смещаются
    """
    #T - количество столетий, прошедших с эпохи J2000
    T = (jd - 2451545.0) / 36525.0
    
    #кглы прецессии (x_p, z, z_p).
    #они определяют, на какой угол отклонилась ось за время T
    x_p = (2306.2181 * T + 0.30188 * T**2 + 0.017998 * T**3) / 3600.0
    y_p = (2306.2181 * T + 1.09468 * T**2 + 0.018203 * T**3) / 3600.0
    z_p = (2004.3109 * T - 0.42665 * T**2 - 0.041833 * T**3) / 3600.0
    
    #перевод в радианы для функций sin/cos
    x_p, y_p, z_p = map(np.radians, [x_p, y_p, z_p])
    '''
    матрица прецессии составляется из трех поворотов
    1 вокруг оси z на угол -x_p
    2 вокруг оси y на угол z_p
    3 вокруг оси z на угол -z
    '''
    Rz1 = np.array([[np.cos(-x_p), -np.sin(-x_p), 0],
                    [np.sin(-x_p),  np.cos(-x_p), 0],
                    [0,              0,             1]])
    
    Ry = np.array([[ np.cos(z_p),  0,  np.sin(z_p)],
                   [ 0,              1,  0],
                   [-np.sin(z_p),  0,  np.cos(z_p)]])
    
    Rz2 = np.array([[np.cos(-z_p),    -np.sin(-z_p),    0],
                    [np.sin(-z_p),     np.cos(-z_p),    0],
                    [0,              0,             1]])
    
    #матрица P переводит из J2000 в текущую эпоху jnow
    #транспонируем
    P = Rz2 @ Ry @ Rz1
    return P.T 

def get_gst(jd):
    """
    вычисляет GMST (Greenwich Mean Sidereal Time ) — звездное время в гринвиче.
    звездное время показывает угол поворота земли относительно звезд(не солнца).
    """
    T = (jd - 2451545.0) / 36525.0
    #формула упрощает угол в градусах на момент jd
    gst = 280.46061837 + 360.98564736629 * (jd - 2451545.0) + 0.000387933 * T**2
    return np.radians(gst % 360)

def horizontal_to_j2000(az, el, dist, lat, lon, jd):
    """
    функция делает понятными для другиx спутников координаты которые мы получаем из точки на земле
    """
    az_r, el_r = np.radians(az), np.radians(el) #сферич коорд
    v_loc = np.array([
        dist * np.cos(el_r) * np.cos(az_r),
        dist * np.cos(el_r) * np.sin(az_r),
        dist * np.sin(el_r)
    ])

    #учет широты
    #поворачиваем локальный вектор так, чтобы ось z стала параллельна оси вращения земли
    lat_r = np.radians(lat)
    R_lat = np.array([
        [-np.sin(lat_r), 0, np.cos(lat_r)],
        [0,              1, 0],
        [np.cos(lat_r),  0, np.sin(lat_r)]
    ])
    v_fixed = R_lat @ v_loc #вектор привязанный к центру земли

    #учет долготы и вращения земли
    #наxодим сектор неба над нами 
    lst = get_gst(jd) + np.radians(lon)
    y_lst = np.array([
        [np.cos(lst), -np.sin(lst), 0],
        [np.sin(lst),  np.cos(lst), 0],
        [0,            0,           1]
    ])
    z_nut = y_lst @ v_fixed

    #учет нутации 
    N_inv = get_nutation_matrix(jd)
    v_prec = N_inv @ z_nut

    #учет прецессии (jnow -> J2000)
    #исправляем координаты на величину изменения земной оси с 2000 года
    P_inv = get_precession_matrix(jd)
    v_j2000 = P_inv @v_prec
    
    return v_j2000

# пример данных:
# JD для 26г ~ 2461041.5
v_res = horizontal_to_j2000(az=45, el=30, dist=1000, lat=55.75, lon=37.62, jd=22461041.5) #для примера значения

print(f"Вектор в системе J2000 (X, Y, Z): {v_res}")










def j2000_to_geo(v_j2000, lat, lon, jd): #lat - широта, lon - долгота 
    """
    Преобразует вектор из инерциальной системы J2000 
    в горизонтальную систему координат (Az, El, Dist)
    """
    # 1. Обратная прецессия (J2000 -> Jnow)
    P = get_precession_matrix(jd)
    # Так как в функции get_precession_matrix возвращается P.T, 
    # для обратного хода берем саму матрицу P (транспонируем результат функции)
    v_prec = P @ v_j2000

    # 2. Обратная нутация (Средняя эпоха -> Истинная эпоха)
    N = get_nutation_matrix(jd)
    z_nut = N.T @ v_prec

    # 3. Обратный учет вращения Земли и долготы
    lst = get_gst(jd) + np.radians(lon)
    y_lst = np.array([
        [np.cos(lst), -np.sin(lst), 0],
        [np.sin(lst),  np.cos(lst), 0],
        [0,            0,           1]
    ])
    v_fixed = y_lst.T @ z_nut

    # 4. Обратный учет широты
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

# Пример проверки:
az, el, d = j2000_to_horizontal(v_res, lat=55.75, lon=37.62, jd=22461041.5)
print(f"Az: {az:.2f}, El: {el:.2f}, Dist: {d:.2f}")
