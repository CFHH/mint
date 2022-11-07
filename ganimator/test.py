import os
import numpy as np
import math
import random
import torch
from tools.Quaternions import Quaternions
from tools.transforms import quat2euler

"""
rotation_deg = [-18.615620, -84.589119, -20.538258]

quat = Quaternions.from_euler(np.radians(rotation_deg), world=False)
print(quat.qs)

rot_rad = quat.euler()
print(np.degrees(rot_rad))

quat2 = Quaternions.from_euler(rot_rad, world=False)
print(quat2.qs)
"""

rotation1 = [-18.615620, -84.589119, -20.538258]
rotation2 = [-168.822006, -80.244133, -169.368124]

quat1 = Quaternions.from_euler(np.radians(rotation1), world=False)
print(quat1.qs)
quat2 = Quaternions.from_euler(np.radians(rotation2), world=False)
print(quat2.qs)

q00 = Quaternions.slerp(quat1, quat2, 0.0)
print(q00.qs)

q25 = Quaternions.slerp(quat1, quat2, 0.25)
print(q25.qs)

q50 = Quaternions.slerp(quat1, quat2, 0.5)
print(q50.qs)

q75 = Quaternions.slerp(quat1, quat2, 0.75)
print(q75.qs)

q100 = Quaternions.slerp(quat1, quat2, 1.0)
print(q100.qs)