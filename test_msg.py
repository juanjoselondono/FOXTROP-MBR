import numpy as np
from geometry_msgs.msg import PoseStamped
msg = PoseStamped()
try:
    msg.pose.position.x = np.float64(1.0)
    print("Success")
except Exception as e:
    print(f"Failed: {type(e).__name__}: {e}")
