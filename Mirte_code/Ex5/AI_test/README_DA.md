# Fletning af kollegaens MCL og selflocalize

Læs KOLLEGA_FLETNING_DA.md for de nyeste ændringer, marker-ID-forskellen og den eksperimentelle augmented MCL. selflocalize.py er en ny tilpasset viewer. Den glidende navigation bevares, og augmented recovery er som standard slået fra.

# Opdatering fra opgave 1–4

Læs GENBRUG_DA.md først. Pakken anvender nu driverfaktorerne 2.38/2.38 fra ContinuousDrive.py og indeholder calibrate_drive.py samt capture_camera.py. Motorresponsen ved lave hastigheder er stadig uverificeret. De gamle kalibreringstal er ikke automatisk aktiveret som fysiske gains. Gainværdierne er neutrale og skal måles efter opsætning af modifierne.

# MIRTE med glidende kørsel

Brug denne samlede pakke i stedet for den tidligere MIRTE_fixed-pakke. De 19 oprindelige moduler og rettelserne til MCL, kameramålinger, kortgeometri og RRT er bevaret. Kørslen i opgave 5 og RRT's path follower er nu løbende hastighedsstyring.

## Bevægelsen

Robotten sender `drive(v,w,None,blocking=False,interrupt=True)` med nye hastigheder cirka hver 0,15 s. Den eksisterende motorhastighed fortsætter under kameraaflæsning, positionsopdatering og ventetid. Kontrolopdateringer udløser ingen stop. Lineær hastighed er normalt højst 6 cm/s i opgave 5 og 8 cm/s i RRT. Acceleration og ændringer i drejehastighed begrænses, så kommandoerne ændres gradvist.

MCL beregner en cirkelbue med den tidligere aktive hastighed over den faktisk forløbne monotone tid. Samtidig rotation og fremkørsel modelleres som én bevægelse. Løkken bruger ikke en kunstig minimumstid eller flytter robotten i små afsluttede trin.

Opgave 5 starter med lokalisering før fremkørsel: én sammenhængende søgedrejning efter markørerne og en stationær forfining. Under kørslen drejer robotten langsomt frem og tilbage for at se begge markører, mens den fortsætter fremad. Tæt på kasserne sænkes fremkørslen til 0,8 cm/s under disse synsfeltjusteringer. Når begge markører er nyligt observeret og målet er under 19 cm væk, styres den sidste indkørsel mod midtpunktet. Succes kræver afstand plus to gange positionsspredningen højst 10 cm og begrænset bevægelse siden begge markørmålinger.

Robotten stopper ved mål, sensorfejl, for stor usikkerhed, manglende målinger, en farlig forudsagt kurve eller timeout. En stor indledende retningsfejl eller en skarp rute kan kræve drejning på stedet. Der er ingen rutinemæssige stop mellem målinger eller små translationssegmenter. RRT-søgning kan fortsat standse ved afslutningen af en hel rute for at opbygge et nyt lokalt kort; dens særskilte søgedrejninger ligger før den næste rute.

## Robotdriveren skal understøtte løbende kommandoer

`ku_mirte.py` var ikke blandt de modtagne filer. Den oprindelige path follower brugte allerede ovenstående løbende API, og denne version bruger samme kald. Koden kontrollerer inden motorstart, at driverens argumenter understøtter kaldet. Kontrollen kan ikke bevise, hvordan driveren styrer motorerne internt.

Driveren skal erstatte den aktive hastighed uden at stoppe eller sætte kommandoer i kø. `duration=None` skal holde hastigheden aktiv, til den erstattes. Tidsmodellen antager, at den gamle hastighed fortsætter under indsendelsen og den nye er aktiv, når kaldet returnerer. Hvis driveren virker anderledes, skal `robot_io.py`/MotionTracker tilpasses dens faktiske tids- og kørselssemantik. Ved afvist API skal den konkrete `ku_mirte.py` gennemgås; programmet starter ikke fremkørsel med en inkompatibel driver.

Der er ikke foretaget fysisk robotkørsel. Simulation tester løbende kommandoer, billedaflæsning og driverlatens, men ikke fysisk motorregulering, hjulslip, skjulte forhindringer eller virkelige kameratidsstempler. Kameraobservationer behandles ved modtagelsestidspunktet; reel kameraforsinkelse skal holdes lille eller kompenseres med tidsstempler/odometri i driveren.

## Opsætning og kørsel

Udpak filerne samlet og kontrollér `ex5_config.py`:

- ArUco-ID'er er **1 og 10**, og målt afstand mellem markørcentre er **1,20 m**. Tilpas til jeres opstilling.
- Kamera: **640×480**, brændvidde **609,9 px**, markør **145 mm**, kameraoffset **0,14 m**. Kamera antages vandret.
- Kasser: **0,40×0,25 m**, robotradius **0,22 m**, clearance **0,025 m**. I opgave 5 antages begge frontmarkører at vende mod negativ z, så kassens centrum ligger 0,125 m bag markøren i positiv z. Tilpas `LANDMARK_BOX_CENTER_OFFSETS_M`, hvis opstillingen er anderledes. Kollisionsmodellen bruger en cirkel omkring det faktiske kassecentrum, som omslutter kassen, plus robotradius og positionsusikkerhed.
- Positiv drejehastighed skal være mod venstre. Lineær/vinkelhastighed skal være kalibreret; tidsintegrationen er forventet bevægelse, ikke målt encoderodometri.
- `mirte.sonar['front_left']` og `['front_right']` skal være opdaterede afstande i meter.

```bash
cd MIRTE_merged
python3 check_setup.py
python3 test_project.py
```

Sæt om nødvendigt `KU_MIRTE_PYTHON_PATH` til mappen med jeres eksisterende `ku_mirte.py`. Læs sensorer uden motorstart:

```bash
python3 check_setup.py --robot
python3 geometry_check.py
```

Kør opgave 5:

```bash
python3 run_between_boxes.py
```

Kør særskilt RRT/søgning efter et bestemt ID:

```bash
python3 run_to_box.py
```

Ctrl+C stopper via `finally`. Brug ikke moduler blandet fra gamle og nye pakker.

## Bevarede rettelser

MCL bevarer vægtprioren, også når resampling springes over, og beregner vægtet position, vinkel og usikkerhed. Ingen målbekræftelse baseres på den uniforme startpopulation. De to startmarkører samles i samme rotationsframe. Kortets kendte markører holdes adskilt fra nye observationer; RRT bruger et frosset lokalt kort under en rute.

ArUco-kompatibilitet understøtter OpenCV 4 og 5. Billedopløsning kontrolleres mod kalibreringen. Byte-identiske billeder regnes ikke som friske målinger; dette er konservativ detektion af frosne billeder, ikke en erstatning for kameratidsstempler. LocalMap holder kasser og markører konsistente. Rette segmenter kontrolleres med præcis afstand til orienterede rektangler; kurver kontrolleres med korte kordesegmenter og en ekstra margin.

FrameBuffer kopierer billeder, gridstørrelser er konsistente, vinkler pakkes korrekt, punktmassemodellen håndterer nulafstand, og MirteModel gemmer retningen i hver søgetilstand. Hardwareimport er lazy og tests starter ingen fysisk robot. Den ældre `camera.py` bruger en separat cm/3D-abstraktion og skal ikke levere de plane meterobservationer til MCL.

## Filer og deres rolle

| Filer | Funktion |
| --- | --- |
| `run_between_boxes.py`, `ex5_config.py` | Opgave 5, startlokalisering, glidende navigation og fysisk konfiguration |
| `mcl.py`, `particle.py`, `random_numbers.py` | Partikelfilter og partikel-/støjhåndtering |
| `continuous_control.py`, `robot_io.py` | Tidsintegration, hastighedsbegrænsning, kurvecheck og robotdrivergrænse |
| `aruco_measurements.py`, `aruco_compat.py` | Nye plane markørobservationer, kamerageometri og OpenCV-kompatibilitet |
| `camera.py`, `framebuffer.py` | Ældre kameraabstraktion og trådsikker billedbuffer |
| `local_map.py`, `geometry_utils.py` | Lokalt kassekort, koordinater og præcis kollisionsgeometri |
| `mirte_rrt_smooth.py`, `path_smoothing.py` | RRT, ruteudglatning og fast kort under ruten |
| `path_follower.py` | Løbende hastighedsstyring langs en hel rute med MCL |
| `run_to_box.py`, `between.py` | Søgning efter ID og valg af sikre passager |
| `robot_models.py`, `grid_occ.py` | Bevægelsesmodeller og gitterrepræsentation |
| `geometry_check.py`, `visualize_local_map.py` | Geometrikontrol, kortvisning og JSON |
| `simulate_mcl.py`, `test_project.py`, `check_setup.py` | Virtuel robot, automatiske tests og opsætningskontrol |

## Validering

`validation/` indeholder testlogs fra OpenCV 4 og 5 samt JSON med simulationsresultater. Den virtuelle robot holder v/w aktive mellem kald og bevæger sig også under 30 ms billedaflæsning og 20 ms driverindsendelse. Kameraets synsfelt og markørfladernes synlighed er begrænset. Ideelle motorer og 10 % forskel i fremkørsels-/drejehastighed testes hver med 20 tilfældighedsfrø. Resultater og stopantal fremgår af JSON, herunder forsøg, hvor styringen stopper uden at bekræfte målet.

Kør selv:

```bash
python3 test_project.py
python3 simulate_mcl.py
```

Disse resultater dokumenterer de simulerede scenarier. Den fysiske robot og driverens interne kommandohåndtering skal stadig afprøves.
