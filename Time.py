import numpy as np
from astropy.time import Time
times = ['2025-12-12T00:00:00.123456789', '2007-02-21T00:00:00']
t = Time(times, format='isot', scale='utc')
print('UTC :', t)
print('TT :', t.tt)
print('TAI :', t.tai)

