# Gennemgang af opgave 3–4 og bevægelsestest

Denne opdatering bygger videre på MIRTE_smooth. Den nye motoropsætning er hentet fra jeres ContinuousDrive.py. Den er ikke fysisk efterkalibreret i denne gennemgang. KU_Mirte-driverens implementering er stadig ikke med.

## Hvad hver af de 14 filer bidrager med

| Fil | Fund og beslutning |
|---|---|
| ContinuousDrive.py | Bruger set_driving_modifier(2.38,2.38), fremkørsel 0.15, kurver 0.375 og spin 1.5. Genbrugt motoropsætningen. Dødzonekommentarer er vigtige, men ingen målelog følger med. Sonar-watchdog er en nyttig idé, men dens separate motorstop kan konkurrere med navigationens kommandoer; den kopieres derfor ikke direkte ind. |
| test.py | Noterer ligeud ved 0.35, drift -0.003625, tid 2.7 og venstreskala 0.975. Reference til nye målinger; aktiveres ikke som universel kalibrering. Flere interaktive løkker efterfølges af automatiske cirkelbevægelser, så q fra en løkke stopper ikke nødvendigvis hele programmet. |
| measurements(5).py | Tilbyder manuel afprøvning ved 0.30 og drejning ved 0.7. Skala ændrer drejetiden. Erstattet med en ny måletest, som udfører én valgt manøvre og gemmer måleresultatet. |
| camera(10).py | f=609.9, markør 145 mm og billeder 640x480. Værdier allerede anvendt. Stopafstand 400 er millimeter fra kameraet; det er ikke 400 meter eller en robotcentreret afstand. Finite nonblocking kommandoer i en hurtig løkke, manglende finally-stop og manglende billedvalidering gør direkte genbrug uhensigtsmæssigt. |
| cameraCalibrations_v4(5).py | Modtager ROS2 compressed images med sensor-QoS og kopierer billeder. Programmet optager billeder, men beregner ikke kamerakalibrering. Ny capture_camera.py bevarer ideen og afviser gemning uden ny/recent modtagelse, kontrollerer brugbare header-tidsstempler og overskriver ikke eksisterende billeder. |
| map(2).py | Markørkort i kameraets x/z og millimeter; ingen robotoffset eller kassegeometri. Den nuværende måle-/kortkode har disse korrektioner og bevares. |
| mapplot(3).py | PNG- og JSON-gemning af markørkort er allerede dækket af visualize_local_map. Det gamle format er kamera-mm; det nye er robotcentreret og i meter. Ingen ny fysisk kalibrering. |
| rotation(2).py | Visualiserer markørens akser og dybdeoffset. Bekræfter den brugte millimeterkonvertering og negative lokale Z. Som helt program returnerer get_map_from_mirte forskellig struktur med og uden fund; while-løkkens tuple-udpakning kan derfor fejle. Ingen ny drejekalibrering trods filnavnet. |
| local_map(20261007-060926).py | Kameraoffset +0.14 m og kasseoffset R@[0,0,-125] mm. Allerede bevaret. Den vedhæftede version bruger cirkulære forhindringer og kassecentre som landmarks; den nyere version adskiller markørcentre til MCL fra orienterede kasser til kollisionscheck. |
| between(10).py | Har landmark_dict, som allerede findes i den rettede pakke. Ved netop to markører eller ønskede ID'er returneres midtpunkt uden passagebreddecheck. Det ville fjerne de nyere geometrikontroller, så versionen bruges ikke som erstatning. |
| mirte_rrt(10).py | Har Execute_path med v=0.30, angular correction=-0.0354 og tidsfaktor 1.05 venstre/1.10 højre ved 0.7. Disse er referenceværdier. Programmet opdaterer sin pose til waypointet uden at måle ankomsten og bruger stop/start mellem segmenter. Den glidende MCL-styring bevares. |
| robot_models(10).py | Matcher den tidligere uploadede original. Rettelserne til nulafstand og retning gemt i hver state er allerede med. |
| grid_occ(10).py | Matcher den tidligere uploadede original. Gitterrettelserne er allerede med. |
| visualize_local_map(9).py | Matcher den tidligere uploadede original. Den nye udgave viser orienterede kasser, geometri og clearance, som bør bevares. |

## Ændringer i den nye pakke

- robot_io.KU_Mirte anvender motorfaktorer 2.38/2.38 én gang ved oprettelse. Mangler driverfunktionen, stoppes robotten og en konkret fejl vises.
- ex5_config samler motorfaktorer, fysisk respons og kameraets fx/fy/cx/cy/distortion. Kameraværdierne er uændrede; nul-distortion er stadig en antagelse.
- set_velocity oversætter fysisk ønsket hastighed til driverkommando, så MCL fortsat integrerer den forventede fysiske bevægelse. Lineær gain, separate venstre-/højregains og drift pr. meter har neutrale værdier, indtil de er målt i den aktuelle opsætning.
- RRT's søgedrejninger og den ældre move-hjælper bruger også denne oversættelse.
- calibrate_drive.py giver én valgt fysisk test med --run, løbende sonarcheck og afsluttende stop. Uden --run er den offline. Distance, drejevinkel og eventuel drift indtastes efter stop og gemmes i JSONL.
- capture_camera.py giver opdateret billedoptagelse med samme ROS-topic og sensor-QoS som V4. Import starter hverken ROS eller motorer.
- check_setup viser motorfaktorer og om lavhastighedsområdet er fysisk verificeret.

Der indføres ingen skjult minimumshastighed eller pulserende motorstyring. Kontrolopdateringerne stopper ikke robotten mellem målinger. DRIVE_LOW_SPEED_VERIFIED er en dokumentationsmarkering; den er ikke en automatisk fysisk detektor eller en kørselsspærre.

## Kalibreringsberegninger og hvorfor gamle tal ikke kombineres

For ligeud gælder v_målt=d/t og k_v=v_målt/v_kommando.

Hvis test.py's 2.7 sekunder faktisk svarer til en målt meter ved kommando 0.35:

    v_målt = 1 / 2.7 = 0.370370 m/s
    k_v = 0.370370 / 0.35 = 1.058201

Firkanttestens tidligere 2.55 sekunder ville give 1.120448. Det er forskellige værdier. Ingen målelog afgør, hvilken der passer til den aktuelle opsætning, og ingen af testene kalder selv set_driving_modifier.

Ved amplitudeskala 0.975 og tid pi/2 er den nominelle drejning 0.975*90 = 87.75 grader. Hvis den fysiske drejning var 90 grader, er k_w=90/87.75=1.025641.

RRT-testen ændrer derimod TIDEN. Hvis dens drejning er præcis, giver tidsfaktor 1.05 en venstregain 1/1.05=0.952381 og tidsfaktor 1.10 en højregain 1/1.10=0.909091. Det er ikke det samme som at gange angular speed med 1.05/1.10. Disse gains ved 0.7 kan ikke uden måling overføres til 0.30 eller 1.5.

Testfilens korrektion -0.003625 i 2.7 sekunder giver -0.0097875 rad = -0.560783 grader nominelt. RRT-korrektionen -0.0354 ved v=0.30 giver over en nominelt kørt meter -0.0354*(1/0.30)=-0.118 rad=-6.760902 grader. Forskellen er stor; vi aktiverer ikke en af dem vilkårligt.

ContinuousDrive's nominelle kurveradius er 0.15/0.375 = 0.40 m. Radius gælder kun fysisk, hvis fremkørsel og rotation har passende respons.

Den nye kommandokonvertering bruger følgende MODEL, som skal måles:

    faktisk_v = k_v * kommando_v
    faktisk_w = k_w * kommando_w + b * faktisk_v

For fysisk mål (v,w) sendes derfor:

    kommando_v = v / k_v
    kommando_w = (w - b*v) / k_w

k_w vælges efter fortegnet på den korrigerede drejekommando. b er målt drift i radianer pr. meter. Det er målte systemparametre, ikke en identitet for en vilkårlig motor. Gain=1 og b=0 er stadig uprøvede neutralantagelser.

## Lav hastighed er fortsat uafklaret

ContinuousDrive kommenterer en fremkørselsdødzone omkring 0.12 og behov for spin omkring 1.5. MIRTE_updated bevarer de glidende mål på højst 0.06 m/s, 0.008 m/s under tæt synsfeltjustering og scan 0.30 rad/s. De værdier er IKKE bevist stabile på robotten. At sætte motorfaktorerne er ikke et bevis på, at dødzoneproblemet er løst.

Inden navigation skal de lave områder måles på samme robot, underlag, batteritilstand og modifieropsætning. Hvis robotten ikke kan bevæge sig stabilt langsomt, skal motorstyringen ændres eller navigationsforløbet tilpasses den faktisk mulige hastighed. En gain kan ikke gøre en fysisk dødzone lineær, og en højere kommando må ikke skjules for bevægelsesmodellen.

## Brug af måletesten

Offline plan, uden hardware:

    python3 calibrate_drive.py --mode linear --speed 0.15 --seconds 2

Fysisk test i frit område, med motorstart:

    python3 calibrate_drive.py --mode linear --speed 0.15 --seconds 2 --run
    python3 calibrate_drive.py --mode left --speed 1.5 --seconds 1 --run
    python3 calibrate_drive.py --mode right --speed 1.5 --seconds 1 --run

Derefter måles også de faktisk ønskede lave hastigheder, fx 0.06 og 0.008 samt scan 0.30. Hvis målt afstand/vinkel er nul, er det et dødzonefund, ikke en brugbar gain. Gentag forsøg og kontrollér, at responsen er tilstrækkelig stabil over det område, hvor en fælles gain anvendes. Fælles gain kan være utilstrækkelig, især på mecanumhjul og ved samtidig rotation/fremkørsel. Programmet ændrer ikke konfigurationen automatisk.

Billedoptagelse, uden motorstyring, i et ROS2-miljø med GUI:

    python3 capture_camera.py --output camera_images --count 10

SPACE gemmer nyeste nyligt modtagne billede; q afslutter. Kendte gentagne/tilbagegående ROS-headerstempler afvises. Hvis headerstempler er nul, kan værktøjet kun kontrollere nye callbacks og modtagelsestid. Opdaterede callbacks kan ikke bevise, at kameraet eksponerer friske billeder. Værktøjet løser ikke nødvendigvis det tidligere problem i selve kamerastreamet. Billederne skal efterfølgende bruges til en egentlig kamerakalibrering; den beregnes ikke her.

## Validering og begrænsninger

Tests og syntetisk robotsimulation er i validation. Den fysiske driver, motorfaktorernes interne effekt, dødzone og ROS-kamerakørslen kan ikke efterprøves i dette miljø. Testene afprøver modifierkald, kommandokonvertering, fysisk tidsmodel, måleberegning, stop på fejl, billedkopier, gemmefriskhed og kamerakonfiguration. De er ikke en ny fysisk kalibrering.

De vedhæftede originaler er ikke ændret. Brug MIRTE_updated samlet; kopier ikke de ældre mellem-/kort-/RRT-versioner oven i den. ku_mirte.py er fortsat næste nødvendige fil til at kontrollere driverens interne semantik.
