import numpy as np

# Test the math
T_vio_0 = np.eye(4)
T_vio_prev = np.eye(4); T_vio_prev[0,3] = 1.0 # prev is at x=1
T_vio_curr = np.eye(4); T_vio_curr[0,3] = 2.0 # curr is at x=2

# We want delta that maps from curr to prev
# x_prev = delta @ x_curr
# origin of curr (0,0,0) should map to (1,0,0) in prev's frame
# Wait. If prev is at x=1 and curr is at x=2, then curr origin is at x=1 in prev's frame.
# So delta should translate by [1, 0, 0].

delta_vio = np.linalg.inv(T_vio_prev) @ T_vio_curr
print("delta_vio translation:", delta_vio[:3,3]) # Should be [1, 0, 0]

T_map2cam_prev = np.linalg.inv(T_vio_prev) @ T_vio_0
T_map2cam_curr = np.linalg.inv(T_vio_curr) @ T_vio_0

# The old buggy delta_vio_map:
buggy = np.linalg.inv(T_map2cam_prev) @ T_map2cam_curr
print("buggy translation:", buggy[:3,3])

# The corrected delta_vio_map:
corrected = T_map2cam_prev @ np.linalg.inv(T_map2cam_curr)
print("corrected translation:", corrected[:3,3])

