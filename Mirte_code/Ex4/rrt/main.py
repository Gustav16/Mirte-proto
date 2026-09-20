#Mirte Proto code main file




def main():

    # 1. Get camera / ArUco detections
    landmarks = ...

    # 2. Build local map
    local_map = ...

    # 3. Create RRT
    rrt = ...

    # 4. Plan
    path = rrt.planning()

    # 5. Execute path on Mirte