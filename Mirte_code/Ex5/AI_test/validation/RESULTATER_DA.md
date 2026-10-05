# Validering af den rettede pakke

Dato: 5. oktober 2026. Ingen fysisk robotkørsel.

- Alle 24 Python-moduler syntaksparset/kompileret.
- 33/33 unittest-tests bestået med OpenCV-contrib 4.10.0.
- 33/33 unittest-tests bestået med OpenCV-contrib 5.0.0.
- 100 stationære MCL-seeds: medianfejl 0.88 cm, største fejl 2.53 cm; ingen over 10 cm.
- 20 kørselsseeds med ideelle motorer, begrænset kameraudsyn og markørflader: 20/20 bekræftede mål, ingen falske succesmeldinger.
- 20 kørselsseeds med translation 10 % kortere og rotation 10 % større: 18/20 bekræftede mål, ingen falske succesmeldinger. To forsøg stoppede uden bekræftet mål.
- De fulde logfiler og simulationsdata ligger ved siden af denne tekst.

Simulationen bruger planare, perfekte range/bearing-målinger, når markørerne er synlige i kørselsforsøgene. Den stationære test tilføjer målefejl på 2 cm standardafvigelse og 1,5 grader. Kameraudsyn følger kalibreringen, og de simulerede frontflader afvises over 75 graders indfaldsvinkel. Bevægelsesforsøg modellerer ingen glidning i translationens retning, ROS-latens, occlusion, sonarstøj eller kamerafejl. Andre tests kontrollerer frosne/identiske billeder, manglende målinger, ugyldig sonar, kortintegration og faktisk syntetisk OpenCV-detektion.

Disse tests viser, at de påviste softwarefejl er rettet i de testede scenarier. De er ikke en fysisk godkendelse af MIRTE eller en garanti for alle startopstillinger. Kontrollér kalibrering, sensorenheder og motor-API som beskrevet i README_DA.md.
