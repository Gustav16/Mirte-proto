# Rettet MIRTE-projekt

Denne pakke indeholder alle 19 oprindelige Python-filer med almindelige modulnavne samt fælles hjælpefiler, tests, simulation og denne vejledning. Brug filerne samlet i én mappe. Bland ikke den nye MCL med gamle versioner af path_follower eller mirte_rrt_smooth.

## Hvad der er rettet

- MCL bevarer tidligere vægte, når resampling springes over, og bruger vægtede positions-/vinkelestimater og spredninger.
- Startlokalisering bruger en partikelpopulation foreslået fra støjbehæftede observationer af to kendte markører. Herefter bruges almindelig prediction/correction/resampling. Det erstatter den sparsomme, rent uniforme startpopulation som praktisk startstrategi; løsningen bruger fortsat MCL.
- Ingen translation eller succesmelding uden gyldig startlokalisering. Programmet søger efter begge ID'er ved at dreje på stedet og kan observere dem i forskellige billeder ved samme robotposition.
- Scanrotationer indgår i bevægelsesmodellen. Bearings samles i samme robotframe; cachede målinger genbruges ikke gentagne gange som nye uafhængige målinger.
- Robotten scanner igen under indkørslen. Den sidste scan foretages omkring 22 cm fra målet, hvor markørernes frontflader stadig kan være synlige. Derefter tillades højst 24 cm fremkørsel siden en scan med begge ID'er, og succes kræver et lille modelbaseret fejl-/usikkerhedstjek. Det undgår at kræve læsbare markørflader præcis på linjen mellem kasserne.
- Direkte motorbevægelser er korte, blocking og enten rotation eller lige translation. Dermed afhænger koden ikke af `duration=None`, `interrupt=True` eller den tidligere forkerte løkketidsintegration.
- Ikke-nul STRAIGHT_ANGULAR_BIAS afvises i opgave 5; modellen antager rette translationssegmenter.
- Byte-identiske kamerabilleder regnes ikke som nye målinger. Dette hjælper med at opdage et frosset kamerastream. Det er en konservativ kontrol, ikke et rigtigt kameratidsstempel; en ægte, men helt identisk ny frame bliver også afvist.
- ArUco-detektion virker med OpenCV 4/5 via en kompatibilitetsfil og solvePnP. Billedopløsning skal passe til kalibreringen.
- RRT-grenen bruger et frosset kort af markørcentre til lokal MCL. Robotten starter i denne lokale frame præcis ved `(0,0,0)`. Nye observationer ændrer ikke planlægningskortet.
- `boxes` og `landmarks` i LocalMap opdateres konsistent. Toframe-kortet filtrerer ID'er og ustabile poses og gennemsnitter geometri i samme frame.
- Kollisionscheck af linjesegmenter mod orienterede kasser bruger præcis segment-/rektangelafstand frem for alene diskrete samples. Robotradius og 2,5 cm margin anvendes.
- Passageberegningen bruger nøjagtig mindste rektangelafstand og kontrollerer målpunktets clearance.
- Path follower drejer på stedet og følger hvert segment med korte translationer. Den skærer ikke bevidst hjørner ved at sigte flere segmenter frem.
- Manglende/ugyldig sonar afbryder translation; sensorfejl skjules ikke. Sonar skal være i meter.
- `landmark_dict` er tilføjet, MCL-kald er rettet, og afbrudt kørsel meldes ikke som afsluttet mål.
- Drawing-pose tildeles korrekt efter afsluttede ruter. Ved afbrudt rute nulstilles den næste planhistorik, da faktisk slutpose ikke kendes i tegnerammen.
- Plot og JSON viser/gemmer orienterede kasser og deres geometri. Korttegning viser også robot-clearance.
- FrameBuffer udleverer kopier og returnerer None før første frame. Particle's von-Mises-wrapping, nulafstand i punktmassemodellen, gridcelleantal og gridgenerering er rettet.
- MirteModel gemmer orientering i hver 3D-state og muterer ikke en fælles vinkel mellem RRT-grene. RRT-hovedprogrammet bruger fortsat PointMassModel.
- Hardwareimport er lazy: softwaretests/importer starter ikke ROS eller robotten. Den ældre camera.py er fortsat en separat kameraabstraktion; dens cm/3D-afstand må ikke bruges som den nye MCL's plane meterafstand.

## Før du kører på robotten

Åbn `ex5_config.py` og kontrollér:

1. `LANDMARK_ID_A` og `LANDMARK_ID_B`. De er stadig sat til **1 og 10**.
2. `LANDMARK_DISTANCE_M`. Den er stadig sat til **1.20 m**. Mål mellem ArUco-centrene i gulvplanets koordinater, ikke mellem kassekanter.
3. Markørstørrelse **145 mm**, billeder **640×480**, brændvidde **609.9 pixels** og kameraoffset **0.14 m**. Kamera antages vandret og uden sideforskydning.
4. Kassebredde **0.40 m**, kassedybde **0.25 m** og robotradius **0.22 m**. Kalibreringen er ikke forbedret ved en fysisk måling i denne pakke.
5. At positiv angular speed faktisk drejer mod venstre, samt faktisk lineær-/vinkelhastighed. Hastigheder og tid bruges som forventet odometri; de er ikke encodermålinger.
6. At `front_left` og `front_right` i `mirte.sonar` findes, opdateres og er i meter. Koden stopper ved manglende/ugyldige data.
7. At kamerastreamet opdateres. KU_Mirte's billedmetode giver ikke her et dokumenteret tidsstempel; hashkontrollen opdager identiske, men ikke alle mulige stale billeder.

Der blev ikke kørt fysisk på MIRTE. Robotdriveren `ku_mirte.py` er ikke med i uploaden, så dens enheder og kørselssemantik skal verificeres lokalt. Programmet kræver den API, de oprindelige filer allerede bruger: `drive(v,w,duration,blocking=True)`, `stop()`, `get_image_compressed()` og `sonar`.

## Installation og første kørsel

Udpak ZIP-filen i jeres projektmappe. Mappen `MIRTE_fixed` kan ligge under eksempelvis `~/Mirte-proto/Mirte_code/Ex5/`.

```bash
cd ~/Mirte-proto/Mirte_code/Ex5/MIRTE_fixed
python3 check_setup.py
python3 test_project.py
```

Hvis Python ikke kan finde `ku_mirte.py`, sæt stien til den eksisterende robotdriver. Eksempel, hvis driveren ligger i den viste mappe:

```bash
export KU_MIRTE_PYTHON_PATH="$HOME/Mirte-proto/Mirte/ku_mirte_python"
```

Stien skal være mappen, der faktisk indeholder `ku_mirte.py`. Hjælpefilen søger også i de oprindeligt forventede projektplaceringer. Ingen robotdriver kopieres eller overskrives af denne pakke.

Læs sensorer/geometri uden bevægelse:

```bash
python3 check_setup.py --robot
python3 geometry_check.py
```

Når konfigurationen og sensorerne er verificeret, kør opgave 5:

```bash
python3 run_between_boxes.py
```

Robotten kan begynde at dreje for at søge efter markørerne. Hvis den ikke kan lokalisere sig pålideligt, stopper den uden translation. Der er et begrænset antal scan- og kontroltrin. Ctrl+C udløser stop gennem finally-blokken.

RRT/søgning efter et bestemt ID køres separat:

```bash
python3 run_to_box.py
```

Det program spørger efter et ArUco-ID. Det er ikke hovedprogrammet til opgave 5. Det bygger et nyt kort efter afsluttede/afbrudte lokale ruter. Under en rute stoppes der ved for lang translation uden markørmålinger eller for stort positionsestimat. Kortet dækker kun synlige markørkasser; sonar er backup for andre forhindringer foran robotten.

## Tests og simulering

```bash
python3 test_project.py
python3 simulate_mcl.py
```

Tests bruger unittest og syntetiske sensorer/motorer; de starter ikke en fysisk robot. `simulate_mcl.py` skriver JSON-resultater til terminalen. Gem dem eventuelt med `python3 simulate_mcl.py > simulation_results.json`.

Resultater fra denne pakke findes i `validation/`. De beskriver præcist de testede scenarier, ikke fysisk nøjagtighed på robotten. Den afsluttende fejlcheck er modelbaseret og er ikke en garanti mod ukalibreret systematisk motorfejl. Kalibrering og reelle billeder med markørernes orientering/occlusion skal stadig prøves på robotten. Scan og stop gør kørslen mindre flydende, men giver målinger mellem bevægelserne.

I startopstillinger, hvor markørfladerne ikke kan ses, eller hvor målingerne ikke giver et sikkert estimat, er det korrekte resultat stop uden succesmelding. Det kan kræve en anden fysisk startplacering eller justering af støjparametre efter faktiske målinger. Justér ikke bare tolerancerne for at få en succesmelding.

## Filer

De 19 oprindelige moduler er med. Nye støttefiler: `robot_io.py`, `aruco_compat.py`, `geometry_utils.py`, `check_setup.py`, `test_project.py`, denne vejledning, `requirements.txt` og valideringsresultater. Originalernes download-suffikser er fjernet, så Python-importer passer.

Eksisterende numpy, matplotlib og OpenCV-contrib på robotten kan bruges. `requirements.txt` beskriver Python-afhængighederne til et separat testmiljø; der er ikke installeret eller ændret pakker på jeres robot.
