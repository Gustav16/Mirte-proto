# Denne fil skal køres direkte på Mirte/Raspberry Pi, ikke i WSL på PC'en.
# altså den skal ssh's ind på robotten og køres derfra.

import cv2
import os
import time

try:
    import picamera2
except ImportError:
    print("V5: picamera2 er ikke installeret.")
    print("V5 skal normalt koeres DIREKTE paa Mirte/Raspberry Pi - ikke i WSL paa PC'en.")
    raise SystemExit(1)


def main():
    """
    Direct Picamera2 version based on the course example.

    IMPORTANT:
    Run this directly on Mirte/Raspberry Pi.
    It does not use ROS2.
    """

    image_size = (1640, 1232)
    fps = 30

    save_dir = os.path.dirname(os.path.abspath(__file__))
    image_count = 0

    print("V5: starter Picamera2 direkte paa robotten ...")

    cam = picamera2.Picamera2()

    frame_duration_limit = int(1 / fps * 1_000_000)

    config = cam.create_video_configuration(
        {
            "size": image_size,
            "format": "RGB888",
        },
        controls={
            "FrameDurationLimits": (
                frame_duration_limit,
                frame_duration_limit,
            ),
            "ScalerCrop": (0, 0, 3280, 2464),
        },
        queue=False,
    )

    cam.configure(config)
    cam.start(show_preview=False)

    time.sleep(1)

    print("V5: kamera startet")
    print("V5: SPACE = gem billede, q = stop")

    window_name = "Mirte V5 - Picamera2 - SPACE = gem, q = stop"
    cv2.namedWindow(window_name)

    try:
        while True:

            # Samme grundide som kursets Picamera2-eksempel:
            # hent et nyt frame for hver iteration.
            img = cam.capture_array("main")

            cv2.imshow(window_name, img)

            key = cv2.waitKey(4) & 0xFF

            if key == ord(" "):
                filename = os.path.join(
                    save_dir,
                    f"image_v5_{image_count:04d}.png",
                )

                if cv2.imwrite(filename, img):
                    print(f"V5: billede {image_count} gemt: {filename}")
                    image_count += 1
                else:
                    print(f"V5: FEJL - kunne ikke gemme {filename}")

            elif key == ord("q"):
                print("V5: stopper.")
                break

    except KeyboardInterrupt:
        print("\nV5: stoppet med Ctrl+C.")

    finally:
        cam.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
