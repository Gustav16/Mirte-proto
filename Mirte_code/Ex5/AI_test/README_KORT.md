# MIRTE – hurtig start

Kør kommandoerne på MIRTE fra mappen med de udpakkede projektfiler (`cd MIRTE_merged`, hvis mappen har dette navn). Robotdriveren skal være tilgængelig.

- `python3 run_between_boxes.py` – opgave 5: finder robotpositionen med MCL og kører glidende mod midten mellem to kendte markører.
- `python3 run_to_box.py` – opgave 4: søger efter en markør og bruger lokalt kort og RRT til at planlægge og følge en rute.
- `python3 selflocalize.py` – simulering uden motorstart: viser positionsbestemmelsen ved at gemme et billede af partiklerne.
- `python3 test_project.py` – tester projektet uden fysisk robot.
- `python3 check_setup.py --robot` – kontrollerer opsætning og sensoradgang uden motorstart.

Start kun én kørselsfil ad gangen; den importerer selv de nødvendige moduler. Kontrollér først markør-ID'er og mål i `ex5_config.py` (standard: ID 1 og 10, afstand 1,20 m). Ctrl+C afbryder kørslen og kalder robotstoppet. Projektet er testet i simulation; fysisk kørsel og kalibrering skal stadig verificeres.
