import cv2  # Import the OpenCV library
import time
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../../Mirte/ku_mirte_python'))
from ku_mirte import KU_Mirte

time.sleep(1)  # wait for camera to setup

# ---- Program start ----

mirte = KU_Mirte()

# Gem billeder i samme mappe som denne fil bliver kørt fra
save_dir = os.path.dirname(os.path.abspath(__file__))
image_count = 0

window_name = "Mirte kamera - tryk en tast for at tage billede, 'r' for at stoppe"

try:
    while True:
        img = mirte.get_image_compressed()

        # Vis live billede fra kameraet, så vinduet kan modtage tastetryk
        cv2.imshow(window_name, img)
        key = cv2.waitKey(0) & 0xFF  # vent på et tastetryk (blokerende)

        if key == ord('r'):
            print("Stopper programmet.")
            break

        filename = os.path.join(save_dir, f"image_{image_count:04d}.png")
        cv2.imwrite(filename, img)
        print(f"Billede gemt: {filename}")
        image_count += 1

finally:
    cv2.destroyAllWindows()
    del mirte
