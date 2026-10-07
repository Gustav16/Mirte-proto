# Validering af MIRTE_merged

52/52 tests består med OpenCV 4.10.0 og 5.0.0. Testlogs følger med.

20/20 normale køresimulationer med ideelle motorer og 20/20 med 10 % motorafvigelse bekræfter målet, uden falske succesmeldinger i disse scenarier. Alle forsøg har ét afsluttende stop-kald og ingen (v=0,w=0)-kommandoer efter fremkørslen begynder. Augmented recovery er deaktiveret i disse køresimulationer, som i den normale konfiguration.

Nye tests dækker fast/slow-kvalitetsfald og log-space-underflow, ID-skift, bevarelse af prior uden observation, kortbegrænset sampling, samplingbudget på fuldt blokeret kort, uniforme vægte efter injektion, stop for recovery-usikkerhed, uafhængige snapshots af markørkortet, offline selflocalize med PNG og nulstilling af gammel lokaliseringsstatus ved ny prior.

Den originale uniform_pdf-fejl og bevægelsesmodellens dobbelte rotation er reproduceret uden hardware. Kollegaens model gav (-1,0) for en noiseless 90-graders drejning efterfulgt af én enheds translation i dens +x/+y-konvention.

En ekstra recovery-flytningstest er beskrevet i KOLLEGA_FLETNING_DA.md: recovery nåede ikke under 10 cm fejl efter 100 opdateringer i de tre afprøvede frø, selv om positionsspredningen blev lille. Det er ikke validering af robust automatisk recovery. Derfor er funktionen eksperimentel og FRA som standard.

Ingen fysisk robot, KU_Mirte-driver eller ROS-kamerastream er afprøvet. Motorfaktorernes reelle effekt og lavhastighedsdødzone er stadig uverificerede.
