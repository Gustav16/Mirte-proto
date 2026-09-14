import cv2  # Import the OpenCV library
import time
import os
import sys
 
sys.path.append(os.path.join(os.path.dirname(__file__), '../../Mirte/ku_mirte_python'))
from Mirte.ku_mirte_python.ku_mirte import KU_Mirte
 
time.sleep(1)  # wait for camera to setup
 
# ---- Program start ----
 
mirte = KU_Mirte()
 
# Gem billeder i samme mappe som denne fil bliver kørt fra
save_dir = os.path.dirname(os.path.abspath(__file__))
image_count = 0
 
window_name = "Mirte kamera (live) - tryk en tast for at tage billede, 'r' for at stoppe"
 
try:
    while True:
        # Hent og vis et nyt billede hver gang - dette holder feedet "live"
        img = mirte.get_image_compressed()
        cv2.imshow(window_name, img)
 
        # Ikke-blokerende: venter kun ~1 ms, så løkken bliver ved med at køre
        # og opdatere billedet, i stedet for at fryse indtil et tastetryk
        key = cv2.waitKey(1) & 0xFF
 
        if key == ord('r'):
            print("Stopper programmet.")
            break
 
        # 255/0xFF betyder at der IKKE blev trykket nogen tast i denne omgang
        if key != 0xFF:
            filename = os.path.join(save_dir, f"image_{image_count:04d}.png")
            cv2.imwrite(filename, img)
            print(f"Billede gemt: {filename}")
            image_count += 1
 
finally:
    cv2.destroyAllWindows()
    del mirte