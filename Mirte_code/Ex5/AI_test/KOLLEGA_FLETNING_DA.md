# Fletning af kollegaens tre filer

Pakken bygger videre på MIRTE_updated. Kollegaens arbejde er brugt som grundlag for nye funktioner, men filerne erstatter ikke de eksisterende moduler direkte. Den eksisterende kørsel og dens interfaces er bevaret.

## Hvad der er flettet ind

| Kollegaens fil | Genbrug og tilpasning |
|---|---|
| mcl(4).py | Fast/slow-gennemsnit af målekvalitet og tilfældige nye partikler fra augmented MCL. Implementeret i den eksisterende numpy-MCL med logaritmisk beregning, vægtprior bevaret, grænse for tilfældig andel og endeligt antal forsøg ved kortbegrænset sampling. Funktionen er eksperimentel og tilvalg. |
| selflocalize(2).py | Visning af partikler, vægte, markører og estimeret retning. Ny selflocalize.py bruger projektets MCL og meter/radianer. Default er offline demonstration. --robot læser kameraet stationært og sender ingen drive/stop-kommandoer. --gui åbner valgfri visning. Der gemmes også et PNG. |
| local_map(20261007-073818).py | Kameraoffset +0.14 m og negativ markør-Z var allerede bevaret. Den nye map API tilbyder marker_dict() som kopieret fast markørkort og pose_is_free() til kortbegrænset recovery. RRT bruger disse funktioner. Rektangelgeometri og præcist segmentcheck bevares. |

Motorfaktorerne 2.38/2.38 og måle-/kameraværktøjerne fra den foregående pakke er også med. Den glidende motorløkke er uændret.

## Hvorfor en direkte filudskiftning ville give fejl

1. Kollegaens MCL konstrueres som MCL(prior), hvor prior består af Particle-objekter. Navigationens MCL konstrueres med et fast dict af markørcentre og lagrer partikler numerisk. Objektformaterne og metodekald er forskellige.
2. selflocalize angiver centimeter: ID 1=(0,0), ID 3=(100,0), sigma_range=10, målegrænser 30–500. Vores projekt bruger meter. Korrekte omregninger er 100 cm=1 m, sigma 10 cm=0.10 m og interval 0.30–5.00 m. Kommentarer, der kalder de gamle værdier meter, er misvisende.
3. I kollegaens vinkelkonvention betyder theta=0 bevægelse langs +x. Vores theta=0 betyder frem langs +z, positiv theta er venstre. For samme gulvposition ville vinklerne være relateret ved theta_gammel=theta_ny+pi/2. Derfor kan cos/sin-udtrykkene ikke kopieres direkte ind.
4. Kollegaens uniform_pdf kalder np.asarray(self,z,dtype=float). Det giver TypeError: dtype er angivet både positionelt og ved navn. Funktionens korrekte konvertering ville være np.asarray(z,dtype=float). Den nye MCL bruger ikke denne fejlbehæftede hjælpefunktion.
5. sample_motion_model beregner theta_new=theta+u[1] og bruger derefter theta_new+d_rot_1_est til translation. Det tæller drejningen to gange. Med theta=0, rotation pi/2, translation 1 og nul støj giver den (-1,0), hvor dens egen konvention kræver (0,1). Den nye model bevarer korrekt prediction og cirkelbueintegration for samtidig v/w.
6. Bearing-likelihood er kommenteret ud. Ved stilstand indeholder afstande alene ingen information om robotretningen. Den nye model bruger både range og wrapped bearing.
7. Der er ubegrænsede while-løkker ved kollisionsfri sampling. Et fuldt blokeret kort kan derfor fastlåse programmet. Den tilpassede recovery har et forsøgstal og en tydelig fejl ved manglende fri plads.
8. Gamle vægte kan alle blive nul ved multiplikation af små sandsynligheder. Division med sum(weights) og W_fast/W_slow kan så give NaN. Logaritmiske likelihoods og kvalitetstal undgår almindelig underflow.
9. Resamplede kopier beholder den valgte partikels gamle likelihood som vægt. Et efterfølgende vægtet estimat kan derfor tælle evidensen igen. I den nye MCL nulstilles resamplede vægte til 1/N.
10. selflocalize sætter bevægelsestallene direkte som u=[velocity,angular_velocity] uden integration med dt og uden at sende tilsvarende motorbevægelse. Tastetryk flytter derfor filteret uden at flytte robotten. Den nye stationære robotvisning gør ikke dette; demoens syntetiske bevægelse og filterprediction følges ad.
11. Kameraabstraktionen i den gamle selflocalize giver cm og 3D-distance. Ny MCL bruger den plane robotcentrerede afstand i meter. De gamle målinger skal ikke sendes direkte til den nye MCL.
12. Kollegaens LocalMap repræsenterer kassecentre som cirkler, mens det kendte MCL-kort beskriver ArUco-centre. Den nye version holder punkttyperne adskilt og bevarer den eksisterende orienterede kassegeometri.

## Augmented MCL: hvad der er bevaret og hvad der er ændret

Kollegaens idé er at reagere på faldende målekvalitet. To glidende gennemsnit opdateres:

    w_fast <- w_fast + alpha_fast * (w_avg - w_fast)
    w_slow <- w_slow + alpha_slow * (w_avg - w_slow)
    p_random = max(0, 1 - w_fast / w_slow)

Vi bruger alpha_fast=0.10 og alpha_slow=0.001 som algoritmeparametre fra kollegaens kode. De er ikke fysiske kalibreringer. p_random begrænses til højst 0.10, og nye partikler erstatter tilfældige resamplede partikler.

Eksempel: Begge gennemsnit er 1, hvorefter den nye målekvalitet er praktisk talt nul:

    w_fast = (1-0.10)*1 + 0.10*0 = 0.90
    w_slow = (1-0.001)*1 + 0.001*0 = 0.999
    p_random = 1 - 0.90/0.999 = 0.099099...

Det giver cirka 9.91 % tilfældige partikler i forventning; det faktiske antal er tilfældigt. Beregningerne udføres i log-space med logaddexp, så underflow ikke gør forholdet til 0/0.

I vores vægtede filter bruges predictive evidens sum_i(prior_weight_i * likelihood_i). Kollegaens simple middelværdi er specialtilfældet, hvor vægtene er ens efter resampling. Range og bearing indgår begge i likelihood.

Kvalitetstal sammenlignes kun for samme sæt observerede ID'er. Hvis et ID forsvinder fra synsfeltet, nulstilles referencegennemsnittet for det nye sæt. En mindre synlig markørmængde skal ikke i sig selv udløse global recovery. Ingen observation ændrer ikke vægtprioren eller kvalitetstallene.

Efter en faktisk injektion nulstilles fast/slow-referencen. Dette undgår vedvarende injektion fra én tidligere dårlig måling. Kortbegrænset sampling anvender den eksisterende pose_is_free-funktion og en endelig budgetgrænse. RRT's recovery-bounds svarer til den faste lokale map frame.

Funktionen er slået FRA som standard: AUGMENTED_MCL_ENABLED=False. Den er integreret og testet som mekanisme, men ikke dokumenteret som robust automatisk genlokalisering. Køreprogrammet fortsætter ikke blot, fordi recovery er slået til; det normale usikkerhedsstop bevares.

En ekstra stationær flytningstest på tre tilfældighedsfrø gav efter 100 støjfri billedopdateringer positionsfejl på cirka 0.118, 0.143 og 0.161 m. Der blev indsat henholdsvis 282, 305 og 301 partikler. Spredningschecket alene var tilfreds trods positionsfejlen. Dette er et konkret eksempel på, at en smal partikelpopulation ikke garanterer korrekt position efter flytning. Derfor aktiveres recovery ikke automatisk i navigationen, og de 40 normale køresimulationer validerer IKKE denne flytningssituation.

## Markør-ID'er: kontrollér jeres faktiske opstilling

Kollegaens eksempel bruger ID 1 og 3 med afstand 1.00 m. Vores hidtidige konfiguration bruger ID 1 og 10 med afstand 1.20 m. Det kan være et nyt fysisk setup eller et eksempel fra øvelsen; koden beviser ikke hvilket.

Derfor ændres ID'er og afstand IKKE automatisk. Hvis I faktisk har ID 1 og 3 med målt markørcenterafstand 1 m, ret:

    LANDMARK_ID_A = 1
    LANDMARK_ID_B = 3
    LANDMARK_DISTANCE_M = 1.00

Brug afstanden mellem markørcentre, og kontrollér også kasseorientering og kameraoffset. Alle hovedprogrammer og den nye selflocalize henter disse værdier fra samme config.

## Kørsel og test

Offline demonstration uden robot eller ROS:

    python3 selflocalize.py --frames 100

Offline visning af den eksperimentelle augmented funktion:

    python3 selflocalize.py --augmented --gui --frames 100

Læs kameraet på en STATIONÆR robot, uden navigationskommandoer:

    python3 selflocalize.py --robot --gui --frames 200

I --robot skal begge konfigurerede markører kunne ses i en stationær frame for initialisering. Denne viewer foretager ingen søgedrejning og ingen tastaturstyret robotkørsel. Den antager, at robotten står stille og ikke køres af et andet program. Posegrafen gemmes som selflocalize_particles.png, eller til stien angivet med --output. --robot opretter den normale KU_Mirte og anvender derfor den konfigurerede driveropsætning; den bør ikke køres samtidig med en aktiv motorstyring.

Navigation mellem kasser køres fortsat med:

    python3 run_between_boxes.py

52 automatiske tests, softwarelogs og JSON-simulationer følger med i validation. Der er ikke foretaget fysisk robot- eller drivertest. Motorernes lavhastighedsdødzone er stadig uafklaret som beskrevet i GENBRUG_DA.md. De tre uploadede originaler er bevaret; pakken indeholder den samlede tilpassede version.
