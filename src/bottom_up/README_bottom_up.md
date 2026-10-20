# Verifica e raffinamento bottom-up delle gerarchie di obiettivi

## Finalità e relazione con il top-down

La pipeline bottom-up revisiona la gerarchia di obiettivi già prodotta dal top-down. Assume come ingresso la descrizione del progetto, gli obiettivi di alto livello e le rispettive decomposizioni, quindi verifica quali elementi conservare, correggere, rimuovere o integrare. Il risultato viene salvato separatamente dalla baseline: la revisione utilizza l’estrazione iniziale senza modificarne gli artefatti o rieseguire l’intero procedimento top-down.

Un High-Level Goal, indicato come HLG, rappresenta un risultato funzionale perseguito da un attore e comprende nome, descrizione e attore associato. Un Low-Level Goal, indicato come LLG, descrive un’interazione concreta o un effetto del sistema che contribuisce a realizzare quel risultato. Il branch è l’unità di revisione, composta da un HLG e dai suoi LLG. Il metodo distingue quindi il risultato desiderato dalle operazioni che lo realizzano.

La revisione affronta in particolare la sovrapproduzione e la frammentazione degli HLG. Un’attività può essere documentata e tuttavia non richiedere un padre autonomo, perché ripete un altro obiettivo o rappresenta soltanto un passo operativo. Il criterio perseguito è una gerarchia di risultati funzionali riconoscibili, con responsabilità e perimetri chiari. La riduzione degli HLG è una conseguenza della selezione motivata degli elementi, non un vincolo sul numero finale di obiettivi.

Il passaggio successivo al top-down è utile perché dispone della gerarchia già costruita e delle sue decomposizioni. Anche quando la generazione iniziale usa reflection e considera più HLG, la produzione dei padri precede quella dei figli. Il bottom-up può invece osservare insieme i rapporti tra padri e l’intenzione espressa dalle decomposizioni effettive. I principi GORE rimangono gli stessi, mentre cambiano il materiale disponibile e il compito: costruire una rappresentazione iniziale lascia posto alla revisione di una struttura esplicita.

Questo assetto offre un meccanismo concreto per migliorare precisione, organizzazione e copertura: eliminare elementi eccedenti, correggere decomposizioni incoerenti e cercare risultati documentati ancora assenti. L’entità del miglioramento viene misurata nella valutazione sperimentale separata. La ground truth non è fornita ai modelli come risposta attesa; nel notebook identifica i dataset previsti e viene utilizzata successivamente per valutarli. La fonte delle decisioni del ciclo è la descrizione ricevuta dalla baseline.

## Ingresso e organizzazione di un’iterazione

Il notebook legge direttamente `structuredHighLevelGoals` e `structuredLowLevelGoals` dagli output top-down. Ogni LLG incorpora il proprio padre nel campo `high_level_associated`; l’associazione richiede la corrispondenza esatta con un HLG iniziale, considerando nome, descrizione e attore. Un riferimento incoerente interrompe l’elaborazione del dataset prima delle chiamate ai modelli. Il caricamento non corregge associazioni per similarità e non sostituisce attori: rende esplicita la precondizione strutturale necessaria per sapere quali figli siano sottoposti a revisione.

Ogni iterazione ricostruisce e rivaluta tutti i branch presenti, compresi quelli confermati nelle iterazioni precedenti. La conferma descrive l’esito di un passaggio e può essere revocata da un nuovo giudizio. Questa scelta consente di individuare duplicazioni o difetti anche in elementi già approvati, alla luce della gerarchia aggiornata.

Il ciclo comprende la ricostruzione dell’intenzione dei figli, la valutazione HLG, l’eventuale controllo LLG e l’applicazione delle modifiche ammissibili. L’analisi globale della completezza viene attivata quando lo stato dei branch soddisfa le condizioni previste. I giudizi sui branch sono raccolti prima di applicare le correzioni, così da riferire le decisioni del passaggio a uno stesso stato. Le modifiche strutturali richiedono un’ulteriore iterazione prima della convergenza.

## Ricostruzione HLG′ e significato del confronto inverso

Per ogni branch non vuoto, un modello ricostruisce dai figli un obiettivo diagnostico HLG′. Il ricostruttore riceve soltanto nomi e descrizioni degli LLG, con etichette locali. Non riceve il padre originale, la fonte o gli altri branch. Deve sintetizzare tutte le capacità presenti, comprese quelle eventualmente estranee al perimetro del padre, senza nasconderle dietro una formulazione generica.

HLG′ esprime l’intenzione riconoscibile nella decomposizione; l’HLG originale esprime il risultato che la decomposizione dovrebbe realizzare. Il confronto tra i due fornisce un’indicazione sulla coerenza tra livelli. Se il padre viene mantenuto e la ricostruzione è equivalente, nel percorso ordinario il branch può essere confermato senza una seconda verifica diretta dei figli. Una relazione non equivalente attiva invece il critic LLG, che accerta il problema sugli oggetti effettivi. Per un branch privo di figli la chiamata generativa viene evitata e la relazione è dichiarata non ricostruibile; l’assenza di figli non determina automaticamente la rimozione del padre.

La ricostruzione è una sintesi fallibile. Se un padre documentato richiede creazione, modifica e cancellazione di una prenotazione, mentre HLG′ esprime soltanto la creazione, occorre verificare se le altre capacità siano davvero assenti dai figli o siano state omesse dalla sintesi. Il padre valido non deve essere ristretto soltanto per adattarlo alla ricostruzione. Ai critic viene trasmesso il testo di HLG′, senza la motivazione del ricostruttore, affinché elaborino il proprio giudizio.

Il solo confronto con HLG′ non esaurisce però la revisione degli HLG. Gli LLG iniziali derivano dai padri top-down e possono riprodurne la frammentazione: un padre troppo operativo può generare figli coerenti e un HLG′ equivalente. Analogamente, due padri duplicati possono avere ciascuno una decomposizione internamente corretta. L’equivalenza rende osservabile la coerenza del branch, ma non basta a stabilire l’autonomia del padre nell’insieme. Questa è la motivazione della valutazione diretta degli HLG accanto alla diagnosi inversa: permette di correggere difetti che un controllo concentrato sull’equivalenza può lasciare invariati.

## Critic 1: revisione globale degli HLG dopo la ricostruzione

La ricostruzione HLG′ precede sempre il primo critic. Ogni modello riceve poi una sola richiesta con fonte, intera lista degli HLG originali e tutte le ricostruzioni. Il compito rimane nel Critic 1: non viene introdotto un valutatore aggiuntivo. Cambia l’unità del giudizio, che passa dal singolo branch considerato ripetutamente alla struttura completa, con una decisione per ciascun identificativo.

Il prompt distingue due parti all’interno della stessa risposta. Prima confronta HLG′ e HLG rispetto alla descrizione: il modello deve distinguere un possibile difetto dei figli da un difetto del padre, senza restringere un originale documentato per adattarlo a una ricostruzione incompleta. Poi decide sull’HLG, considerando autonomia, granularità, supporto e responsabilità, oltre a duplicati e coperture sull’intera lista. Una corrispondenza di HLG′ con un sibling non basta a eliminare un padre autonomo; una ricostruzione equivalente non rende autonomo un padre ridondante o troppo operativo. L’ordine è richiesto nello stesso messaggio, non realizzato mediante chiamate separate.

| Parte della risposta del Critic 1 | Oggetto del giudizio | Informazioni restituite |
| --- | --- | --- |
| Diagnosi delle ricostruzioni | Relazione di ciascun HLG′ con il padre originale e gli altri HLG | Per ogni branch: relazione semantica, motivazione ed eventuale identificativo del sibling corrispondente |
| Decisioni strutturali | Organizzazione dell’intera lista dei padri rispetto alla fonte | Per ogni HLG: azione, motivazione, giustificazione dell’autonomia o subordinazione ed eventuali riferimenti di rimozione o consolidamento |

### Autonomia, rimozioni e consolidamento

Per mantenere un HLG il modello deve spiegare quale risultato autonomo esso rappresenti e cosa lo distingua dagli altri. Definirlo documentato, funzionale o coerente non basta: occorre chiarire quale risultato indipendente verrebbe perso eliminando il padre e perché le sue capacità non siano soltanto LLG di un flusso più ampio. I quattro esempi mostrano liste complete con tutti i relativi giudizi, comprese decisioni di mantenimento per risultati simili ma realmente distinti.

La ridondanza completa richiede un altro HLG dello stesso attore che copra l’intero risultato. Il modello indica ID e nome esatti del superstite; il branch corrente non può coprire se stesso. Per duplicati equivalenti vengono preferiti il rappresentante già confermato, poi quello più fedele e completo rispetto alla fonte e, a parità, l’ordine del registro. Le conferme precedenti restano rivedibili. Per coperture asimmetriche viene favorito il risultato documentato più ampio rispetto al frammento operativo.

Un parent che descrive bozze, consensi, controlli sui dati o configurazioni può essere una funzione interna al risultato desiderato. In tal caso il critic può proporre `OVER_SPECIFIC` anche se non esiste un HLG coprente. Il branch e i suoi figli vengono rimossi senza sostituzione immediata; la discovery successiva valuta se serva un parent più ampio. Restano valide le rimozioni `UNSUPPORTED` per intenzioni non documentate e `NON_FUNCTIONAL` per puri vincoli senza nucleo funzionale.

Per altri difetti recuperabili è disponibile `REWRITE`. Il consolidamento usa la stessa azione, indicando un branch coordinatore e tutti gli altri membri dello stesso attore che frammentano un unico risultato. La sostituzione viene applicata soltanto con un nuovo padre e figli accettati. La generalizzazione deve preservare un attore, uno scopo e un perimetro riconoscibili: non si impone un numero finale di HLG e non si accorpano intenzioni indipendenti soltanto per ridurre la lista.

### Diagnosi HLG′ indipendente dall’azione strutturale

Nella prima parte lo stesso modello classifica ogni ricostruzione come equivalente, parzialmente sovrapposta, corrispondente a un altro HLG, differente o non ricostruibile. `MATCHES_SIBLING` richiede l’identificativo di un altro branch dello stesso attore; `NOT_RECONSTRUCTABLE` si usa quando mancano i figli. Il modello non riceve le liste LLG effettive, che spettano al Critic 2. Il codice valida la diagnosi prima della decisione strutturale, senza rendere la validità dell’una dipendente dall’altra.

Decisione sul padre e diagnosi sono validate separatamente. Una rimozione strutturale rimane valida anche con diagnosi `EQUIVALENT`, o con una diagnosi mancante o invalida: l’evidenza per rimuovere deve provenire dal confronto dei padri e dalla fonte. Viceversa, una corrispondenza di HLG′ a un altro goal non basta a eliminare un originale autonomo: può indicare figli associati al padre sbagliato.

Per un padre mantenuto sono necessarie almeno due diagnosi valide. Se almeno una delle diagnosi valide è non equivalente, si richiede la verifica diretta LLG. Se le diagnosi valide sono equivalenti, il branch segue il percorso ordinario di conferma; restano le eccezioni dei budget di riscrittura. Una diagnosi insufficiente produce un esito inconclusivo, senza cancellare la decisione strutturale o inventare un’equivalenza. I campi storici della relazione restano leggibili; nei nuovi risultati le diagnosi individuali sono conservate separatamente.

## Critic 2: verifica diretta degli LLG

La valutazione LLG viene richiesta per un padre mantenuto quando HLG′ non è equivalente, la decomposizione è vuota oppure un’ulteriore riscrittura del padre è stata bloccata dal relativo budget. Il normale mantenimento con equivalenza diagnostica e figli presenti evita questo controllo aggiuntivo. Una decisione di rimozione o riscrittura del padre segue invece il proprio percorso di intervento.

Nella politica `post_repair_llg_audit_v30`, una nuova decomposizione accettata dopo rigenerazione, fusione di duplicati, consolidamento o riscrittura del padre deve ricevere una verifica diretta di Critic 2 nel passaggio successivo disponibile, anche con relazione `EQUIVALENT`. L’orchestratore conserva gli ID in `pending_llg_audit_ids`; il giudizio HLG espone `post_repair_llg_audit_required`, score operativo 4 e il motivo `POST_REPAIR_LLG_AUDIT_REQUIRED`, senza alterare la diagnosi. Il rinvio strutturale o un mancato quorum non consumano l’obbligo. Un giudizio LLG aggregato valido lo conclude; un’ulteriore rigenerazione accettata lo riattiva. Candidati respinti e ripristini non introducono un nuovo obbligo. I nuovi campi hanno valori predefiniti per leggere gli artefatti storici.

Questa verifica non rigenera automaticamente i figli: una decisione valida di mantenimento conclude il controllo, mentre un’altra riparazione richiede quorum e budget ordinari. Restano invariati prompt, discovery, rimozioni e rinvio globale in presenza di proposte strutturali. La verifica successiva riguarda i figli riparati, non le decomposizioni iniziali dei nuovi goal di discovery.

Il secondo critic riceve fonte, padre originale completo, figli effettivi e HLG′. Non riceve i giudizi o le motivazioni degli altri critic. Verifica supporto documentale, perimetro, duplicazioni interne e capacità essenziali mancanti. HLG′ fornisce un’indicazione da controllare: la decisione deve fondarsi sugli LLG effettivi e sulla fonte.

Il critic può conservare la collezione o richiederne la rigenerazione. Quest’ultima richiede una diagnosi esplicita, con identificativi locali dei figli problematici e/o capacità essenziali assenti. Il generatore restituisce l’intera decomposizione, preservando le azioni valide, correggendo quelle incoerenti e aggiungendo soltanto capacità documentate. La collezione accettata sostituisce quella precedente; differenze di stile non giustificano da sole una rigenerazione.

## Ensemble, punteggi e decisione aggregata

Ogni compito valutativo è svolto da Ministral, Granite e Gemma nella configurazione locale corrente. I tre nomi indicano i modelli votanti, mentre Critic 1, Critic 2 e discovery indicano compiti distinti. La maggioranza viene calcolata separatamente tra i tre modelli per ogni compito eseguito. Ciascun modello ha un voto con uguale peso; i punteggi non vengono mediati per determinare l’azione vincente.

| Compito | Origine e significato del punteggio | Regola decisionale |
| --- | --- | --- |
| Giudizio HLG | Il codice deriva 5 per mantenimento equivalente ordinario, 4 quando è richiesto il controllo LLG, anche dopo riparazione, e 2 per riscrittura o rimozione | L’azione richiede almeno due voti validi; il punteggio rappresenta il percorso, non una probabilità di correttezza |
| Giudizio LLG | Il modello restituisce 5 per mantenere oppure 0–3 per rigenerare con evidenze del difetto | Il valore 4 è escluso dagli esiti validi; la decisione richiede due voti validi |
| Completezza | Il modello restituisce 4–5 se non rileva lacune oppure 0–3 con richieste di integrazione | La maggioranza riguarda l’assenza o la presenza di lacune |

La scala ammette quindi valori da zero a cinque, con contratti specifici per compito. All’interno degli intervalli LLG e completezza non è prevista una rubrica di gravità per ogni numero. Nel Critic 1 il modello restituisce una lista di azioni motivate e diagnosi HLG′ separate; eventuali score aggiunti alla risposta non sostituiscono la derivazione del codice. Mantenimenti dovuti a limiti operativi sono identificati separatamente e non diventano misure autonome della qualità del padre.

La validazione considera formato, campi obbligatori, coerenza tra azione e punteggio e riferimenti agli oggetti correnti. Il formato richiesto al Critic 1 è un oggetto con campo `decisions`; una lista JSON completa, anche racchiusa in un blocco Markdown, viene normalizzata nello stesso oggetto senza modificare i giudizi. La risposta grezza viene conservata. JSON incompleti o non interpretabili e identificativi duplicati o estranei invalidano la risposta globale. Una decisione mancante o con un difetto locale invalida soltanto il voto su quel branch; le altre decisioni della lista restano utilizzabili. Un voto invalido non diventa un mantenimento e non abbassa il quorum. In assenza di almeno due voti validi concordi, la valutazione è inconclusiva. Le ripetizioni dei voti HLG mantengono lo stesso prompt senza fornire suggerimenti dei peer. I giudizi sono privi di memoria conversazionale e ciascun compito usa quattro esempi few-shot illustrativi, separati dalle annotazioni dei dataset.

Il campo `evaluation_status`, opzionale per gli artefatti storici, distingue nei log e nei trace l’esito dal motivo dell’attesa. `DEFERRED_STRUCTURAL_ACTIONS` indica un controllo rinviato per dare priorità alle modifiche dei padri; `NO_QUORUM_WITH_INVALID_VOTES` segnala un quorum non raggiunto in presenza di risposte invalide; `NO_QUORUM_DISAGREEMENT` indica disaccordo tra voti validi; `INSUFFICIENT_VALID_DIAGNOSES` identifica diagnosi HLG′ insufficienti. Gli esiti completati hanno stato `DECIDED`. Il rinvio strutturale mantiene la decisione operativa inconclusiva per impedire conferme premature, ma non produce da solo il warning `UNRESOLVED_EVALUATION_INCONCLUSIVE`. Queste distinzioni non modificano il quorum o i criteri semantici.

La maggioranza strutturale di mantenimento viene seguita dalla validazione indipendente delle diagnosi: servono almeno due diagnosi valide e qualunque relazione non equivalente tra quelle valide richiede il controllo dei figli. Per le altre azioni la risposta rappresentativa conserva la relativa motivazione e i riferimenti, mentre tutti i voti rimangono nel trace. Due critic possono richiedere la rimozione dello stesso HLG indicando coperture diverse: la maggioranza sulla rimozione è sufficiente e il target rappresentativo deve superare i controlli applicativi. Non è richiesto l’accordo sullo stesso superstite. Per il consolidamento è invece necessario l’accordo sullo stesso gruppo da assorbire, perché gruppi diversi definiscono sostituzioni diverse.

## Applicazione delle modifiche e generazione focalizzata

Le decisioni dei critic producono richieste di intervento, non nuovi goal già pronti. L’orchestratore controlla dipendenze, riferimenti e budget e richiede al generatore candidati focalizzati. Le funzioni di generazione vengono riutilizzate per riparare la gerarchia corrente, senza rieseguire l’estrazione top-down iniziale. Nella configurazione locale Qwen svolge generazione e ricostruzione; LM Studio serve i modelli, caricati e scaricati sequenzialmente per contenere l’occupazione simultanea di memoria.

Le rimozioni ammissibili hanno priorità sulle riscritture. Più duplicati possono essere rimossi nella stessa iterazione senza un limite numerico fisso, purché i target scelti rimangano disponibili e non si producano cancellazioni reciproche. Anche le rimozioni `OVER_SPECIFIC` sono prive di un limite numerico per iterazione. Il limite di due resta per `UNSUPPORTED` e `NON_FUNCTIONAL`. Restano inoltre i limiti di due operazioni di riscrittura e due rigenerazioni LLG ordinarie per passaggio. Se esistono proposte strutturali aggregate da applicare, il Critic 2 e le rigenerazioni LLG ordinarie vengono differiti; la generazione dei figli necessaria a completare un consolidamento o una fusione resta parte dell’intervento strutturale.

Il consolidamento genera un unico padre per il gruppo e una decomposizione completa, sostituendo i membri congiuntamente soltanto dopo l’accettazione. I gruppi sovrapposti non vengono applicati insieme. Per una rimozione ridondante, i figli delle sorgenti e del target sono raccolti nel contesto della rigenerazione del superstite. Se la riparazione viene respinta, i branch precedenti vengono conservati e la rimozione non viene consolidata. Una rimozione `OVER_SPECIFIC` segue invece la politica esplicita di eliminazione senza target e successiva verifica delle lacune.

Le riparazioni ricevono fonte, motivo dell’intervento, ambito richiesto e capacità da preservare. La generazione è zero-shot nella configurazione del notebook e i candidati sono verificati separatamente da Ministral con quattro esempi few-shot. Questo ciclo di reflection usa una scala da zero a dieci, soglia 8,5 e fino a tre tentativi di generazione. Il feedback negativo guida un nuovo candidato; un problema di formato del giudizio comporta la rivalutazione dello stesso candidato. Tale controllo è distinto dalla maggioranza dell’ensemble.

## Discovery e recupero della completezza

La discovery verifica se la gerarchia corrente rappresenta tutti i risultati funzionali autonomi documentati. È obbligatoria, ma viene attivata quando tutti i branch sono confermati nel passaggio oppure hanno esaurito il budget LLG dopo la rivalutazione dell’ultima decomposizione. I branch al limite rimangono irrisolti: l’accesso alla discovery non li approva.

Il compito utilizza fonte, intera collezione HLG e attori noti. Non riceve LLG, HLG′ o verdetti dei critic precedenti. Il modello deve ricondurre i risultati essenziali della fonte a specifici HLG e segnalare soltanto outcome indipendenti non coperti. Un passo mancante entro un risultato già rappresentato riguarda la decomposizione; un titolo generico non dimostra, da solo, di coprire ogni funzionalità vicina.

Una maggioranza favorevole alle integrazioni autorizza la raccolta delle richieste dei modelli favorevoli, raggruppate per attore. La singola proposta può provenire da uno solo di essi: l’accordo riguarda l’esistenza di lacune, non ciascun dettaglio della richiesta. Il generatore può produrre al massimo tre nuovi HLG complessivi per iterazione, entro gli attori già noti, e genera poi le relative decomposizioni. I nuovi branch devono ricevere la normale revisione.

Le integrazioni vengono applicate soltanto in un passaggio senza sostituzioni o rimozioni HLG. Dopo l’eliminazione di un parent troppo operativo, la discovery può quindi intervenire in un’iterazione successiva sullo stato aggiornato, richiedendo un risultato più ampio se necessario. Il recupero è un esito da verificare: non è una sostituzione automatica del goal eliminato. Questa separazione permette al primo critic di semplificare la struttura e alla discovery di giudicare quali intenzioni siano effettivamente rimaste scoperte.

## Rivalutazione, budget e arresto

Ogni branch sopravvissuto riceve nuovi giudizi nell’iterazione successiva. Le conferme precedenti non proteggono da nuove rimozioni o consolidamenti. La verifica dei figli rimane condizionale, mentre la ricostruzione e la valutazione HLG interessano tutti i branch. La reiterazione controlla dunque anche gli effetti dei cambiamenti sugli elementi già presenti.

Per discendenza sono consentite al massimo tre riscritture HLG. Se viene richiesta un’ulteriore riscrittura, il padre è mantenuto per ragioni operative e viene richiesto il controllo LLG. Il raggiungimento del contatore non blocca un distinto giudizio di rimozione. Ogni identificativo di branch dispone inoltre di due rigenerazioni LLG, con rivalutazione dell’ultima collezione prima di dichiarare il budget esaurito.

Un candidato respinto non sostituisce lo stato precedente. Dopo due cicli di riparazione semanticamente respinti per il branch, il codice ne conserva lo stato e registra una conferma forzata. Il branch continua a essere valutato: ulteriori richieste di riscrittura o rigenerazione vengono bloccate da questa politica, mentre una rimozione motivata resta possibile. I problemi infrastrutturali non sono conteggiati come rifiuti semantici. Le integrazioni della discovery hanno contatori dedicati e il ciclo è limitato a venticinque iterazioni nel notebook, cinque come valore predefinito della libreria.

| Stato | Esito |
| --- | --- |
| Tutti i branch confermati secondo le politiche del ciclo, nessuna lacuna e nessuna modifica strutturale da rivalutare | Convergenza operativa |
| Branch o lacune irrisolti senza ulteriori interventi ammessi | Arresto senza convergenza, con warning e conservazione dello stato corrente |
| Limite globale di iterazioni raggiunto | Restituzione della gerarchia corrente e dello stato di esaurimento del budget |

La convergenza va letta insieme a warning e conferme forzate: descrive la conclusione della revisione secondo le regole del sistema. L’approvazione dei modelli, l’arresto operativo e la qualità rispetto alla ground truth sono proprietà distinte, rese osservabili dai risultati persistiti.

## Tracciabilità e osservazioni progettuali

Il [notebook bottom-up](../../notebook/01_pipeline_execution_bottom_up_only.ipynb) legge gli ingressi in `output/top_down/` e salva risultati ed eventuali errori in `output/top_down_bottom_up/`. Le collezioni finali restano distinte dai dati iniziali. I trace in `output/bottom_up_iterations/` conservano giudizi individuali, risposte grezze, interpretazioni strutturate, decisioni aggregate, warning e contatori. Ogni modello salva la risposta completa in `critic1_global.json`; i file per branch distinguono voti `hlg`, `reconstruction` e `llg`. Il trace di iterazione include `critic1_global_votes`, `hlg_structure_summary` e `structural_action_outcomes`, con proposte, modifiche applicate o differite. Questi campi hanno valori predefiniti per leggere anche i trace storici senza reinterpretarne i voti; i checkpoint registrano anche i passaggi non ancora completati. Il modulo `hierarchy_audit` non viene eseguito dalla pipeline corrente.

Una nuova esecuzione sostituisce gli output dei dataset selezionati: confrontare varianti richiede di conservarne preventivamente gli artefatti e identificare baseline, modelli e configurazione. La logica corrente corrisponde a `reconstruction_first_critic1_v31`; gli artefatti v29 conservano il significato della precedente politica. I prompt completi e gli esempi inglesi sono riportati in appendice; il [notebook di valutazione](../../notebook/02_experimental_evaluation_top_down_vs_bottom_up.ipynb) confronta separatamente gli output persistiti senza generare nuovi goal.

Il contributo progettuale consiste nella corrispondenza tra diagnosi e intervento. Il confronto tra padri affronta ridondanza e granularità; HLG′ rende osservabile l’intenzione dei figli; il critic LLG controlla le differenze sui dati effettivi; la discovery cerca risultati mancanti. Il metodo può così intervenire sia sulla sovrapproduzione degli HLG sia sulle incoerenze tra livelli, anziché assumere che un padre sia adeguato solo perché ricostruibile dalla propria decomposizione.

La maggioranza limita la dipendenza della scelta da una singola risposta e i controlli deterministici disciplinano l’applicazione delle proposte. Rimangono possibili errori concordi dei modelli, perché condividono fonte e istruzioni. La verifica condizionale degli LLG riduce le chiamate, ma una ricostruzione equivalente può nascondere un difetto locale. Rivalutare tutti i branch consente giudizi rivedibili, aumentando il costo computazionale. Le rimozioni senza copertura rendono la revisione più incisiva, ma richiedono di osservare se la discovery recuperi le capacità necessarie a un livello appropriato.

Questi compromessi rendono centrale la disponibilità dei trace. L’utilità della pipeline è fornire una revisione strutturata, motivata e ispezionabile, con baseline preservata e separazione dalle risposte attese. Il miglioramento empirico va misurato considerando precisione e copertura insieme alle capacità conservate o perse e ai casi conclusi per limiti operativi. Il risultato metodologico è un processo nel quale le correzioni e le condizioni della loro applicazione possono essere ricostruite e discusse.

## Appendice. Prompt ed esempi dei critic

Questa appendice riporta in inglese il messaggio di sistema e i prompt completi dei tre compiti dell'ensemble, con le istruzioni e i quattro esempi effettivamente inseriti dal codice. La presentazione espansa rende leggibile il testo ricevuto dal modello senza richiedere di interpretare variabili Python. I dati specifici del dataset e del branch sono sostituiti da segnaposto espliciti; non vengono riportati casi di esecuzione o risultati sperimentali.

Il testo delle istruzioni del Critic 1 è scritto direttamente nel suo prompt: gli elementi inseriti dinamicamente sono soltanto gli esempi e i dati del caso. Durante l'esecuzione, una f-string compone queste parti in un solo messaggio utente per ciascuna valutazione. Il modello riceve il testo composto, non espressioni quali `json.dumps(...)` o nomi di costanti. Il messaggio di sistema e quello utente sono inviati insieme e svolgono ruoli complementari: il primo definisce i concetti e il ruolo generale, il secondo specifica il compito e i dati da esaminare.

### Istruzioni di sistema condivise dall'ensemble

Il messaggio di sistema introduce i concetti GORE e il ruolo del valutatore, con un'impostazione coerente con i critic top-down. Le decisioni e i formati restano quelli specifici del bottom-up.

```text
You are a software engineering assistant specialized in Goal-Oriented
Requirements Engineering (GORE).
An actor is the stakeholder responsible for or benefiting from a capability.
A high-level goal (HLG) expresses WHAT outcome the stakeholder needs and WHY,
independently of implementation. A low-level goal (LLG) describes HOW that
outcome is realized through a concrete system interaction or effect.
A documented automatic effect does not require a manual trigger by the actor.

Your job is to critique artifacts extracted by another assistant from the
software description. Use that description as the source of truth and evaluate
only the assigned task. Examples illustrate criteria, not project requirements.
Do not invent features or responsibilities. Return one valid JSON object without
Markdown. In rationale give concise feedback (at most 200 words): source evidence,
comparison and justified decision. Follow the task-specific score rules.
```

Nei prompt seguenti le istruzioni e i quattro esempi sono espansi integralmente: sono il testo inglese composto dal codice, non la f-string usata per costruirlo. Solo i dati del caso restano rappresentati da segnaposto descrittivi in maiuscolo. Durante l'esecuzione questi segnaposto corrispondono alla descrizione del progetto, agli oggetti JSON, agli identificativi ammissibili e alla ricostruzione del branch corrente.

### Critic 1 globale: prompt completo con quattro esempi di lista

```text
### Critic 1: complete-list High-Level Goal review
You are provided with a software description, ALL current original HLGs and
HLG' reconstructed from each branch's children. You do not see actual children.
Perform the following two parts inside this ONE review and return one entry
for EVERY original branch ID. These are two tasks, not two separate model calls.

### PART 1 -- Diagnose HLG' against the ORIGINAL HLG and SOURCE FIRST
For every branch compare the reconstructed intention with the original parent's
name AND description. Use the software description to distinguish a defect in
the parent from a possible defect in its decomposition. HLG' summarizes current
children; it is not a source of requirements and can itself omit a capability.
In reconstruction_diagnosis classify the relationship:
EQUIVALENT=same essential outcome; PARTIAL_OVERLAP=some capabilities differ;
MATCHES_SIBLING=another same-actor outcome (give matches_branch_id);
DIFFERENT=another outcome with no matching sibling;
NOT_RECONSTRUCTABLE=has_children is false (mandatory only in that case).
Use the registry to identify a matching sibling, not to assume the original
must be removed. A source-supported parent with an incomplete or different HLG'
should normally be kept so Critic 2 can inspect actual children. Do not narrow
that parent to fit the reconstruction or infer missing requirements from HLG'.
If the original has a recoverable source-grounded defect, explain that defect
for a rewrite; a reconstruction mismatch alone does not justify rewriting it.

### PART 2 -- Decide the parent action using the SOURCE and COMPLETE REGISTRY
After the diagnosis, assess the original parent's support, actor responsibility,
independent outcome and granularity. Compare it with all other original HLGs
for duplicates and same-actor full coverage. Read names AND descriptions.
Coherence with HLG' does not prove autonomy: a duplicate or workflow step can
have perfectly coherent children. Being documented, user-visible, or phrased as
Enable/Support/Ensure alone does not justify a standalone HLG.
For each original choose:
- KEEP_ORIGINAL_HIGH_LEVEL_GOAL: justify its independently useful outcome and
  its distinction from the other parents in outcome_justification. Preserve a
  supported independent parent when its reconstruction exposes a child problem.
- REMOVE_ORIGINAL_HIGH_LEVEL_GOAL: FULLY_REDUNDANT requires a complete covering
  SAME-ACTOR goal, exact surviving branch ID and name. Preserve the most complete
  documented outcome; for equivalent peers prefer prior confirmation, then registry
  order. CURRENT cannot cover itself. Different responsibilities and partial overlap
  are insufficient. OVER_SPECIFIC removes internal workflow capabilities even
  without a covering parent: drafts, consents, data checks and configuration may
  belong at LLG level. Explain the containing documented outcome; later discovery
  assesses any missing broader parent. UNSUPPORTED removes wholly undocumented
  intentions; NON_FUNCTIONAL removes pure constraints with no functional core.
  All non-redundant removal bases require null covering fields.
- REWRITE_ORIGINAL_HIGH_LEVEL_GOAL: explain a recoverable defect supported by
  the source and provide rewriting_focus. Do not repair incorrect children by
  changing a valid parent. For complementary fragments of ONE outcome, nominate
  one anchor and list every OTHER same-actor member in consolidate_with_branch_ids.
  Retain source members until that transaction succeeds; avoid reciprocal or
  overlapping consolidation proposals. Preserve all documented capabilities.
A MATCHES_SIBLING diagnosis alone cannot delete a source-supported independent
parent. Conversely, structural REMOVE is valid even with diagnosis EQUIVALENT
when source and registry justify redundancy or over-specific granularity.
Keep diagnosis and action independently justified; do not alter one to fit the other.
Earlier confirmations are revisable. Do not invent broader requirements, combine
independent outcomes, or enforce a target HLG count. Several removals are allowed.
No other model votes or previous verdict texts are supplied.

### Response
Return one JSON object with decisions: a list containing every branch ID once.
Copy IDs exactly from the current registry; example IDs are not current IDs.
For each entry write branch_id and reconstruction_diagnosis first, then decision,
rationale and outcome_justification. The diagnosis requires reconstruction_relationship
and rationale. This order expresses the review sequence, not an extra model call.
Add matches_branch_id only for MATCHES_SIBLING. Add rewriting_focus only for
REWRITE; consolidate_with_branch_ids only for consolidation; removal_basis only
for REMOVE; covered_by_branch_id and covered_by_high_level_goal_name only for
FULLY_REDUNDANT. Omit unused optional fields. Do not return scores or routing flags.
Use concise evidence: at most 25 words each for rationale and outcome_justification,
and 15 for the diagnosis. Preserve semantic distinctions, not long reasoning transcripts.

### Four complete-list examples
Four illustrative complete-list reviews, not project requirements.
For every parent, diagnose HLG' first, then justify the action using the source
and registry. A diagnosis is evidence about decomposition, not the final action.

1. VALID PARENTS, INCOMPLETE OR MISASSOCIATED CHILDREN.
Source: Members create, change and cancel equipment reservations and separately
report damage. Registry: a=Manage equipment reservations, covering all three
reservation actions; b=Report equipment damage. Both belong to Member.
HLG' a describes creating reservations only; HLG' b describes reservations
instead of damage. All current IDs: [a,b].
Output: {"decisions":[{"branch_id":"a","reconstruction_diagnosis":{"reconstruction_relationship":"PARTIAL_OVERLAP","rationale":"Creation alone does not reconstruct changes and cancellation."},"decision":"KEEP_ORIGINAL_HIGH_LEVEL_GOAL","rationale":"The source supports the complete parent; inspect children rather than narrowing it to creation.","outcome_justification":"Member obtains and manages reservations; damage reporting serves a different outcome."},{"branch_id":"b","reconstruction_diagnosis":{"reconstruction_relationship":"MATCHES_SIBLING","matches_branch_id":"a","rationale":"The reconstruction expresses reservations, not damage reporting."},"decision":"KEEP_ORIGINAL_HIGH_LEVEL_GOAL","rationale":"Damage reporting is independently documented; the mismatch concerns its children, not the original parent's legitimacy.","outcome_justification":"Member alerts staff to damage; reservation management does not cover that purpose."}]}

2. SOURCE-GROUNDED PARENT REWRITE.
Source: Applicants submit applications and track their status; only staff decide
approval. Registry: a=Submit, track and automatically approve applications,
whose description grants automatic approval to Applicant. HLG' a describes
submission and status tracking. All current IDs: [a].
Output: {"decisions":[{"branch_id":"a","reconstruction_diagnosis":{"reconstruction_relationship":"PARTIAL_OVERLAP","rationale":"The reconstruction omits the original parent's automatic approval capability."},"decision":"REWRITE_ORIGINAL_HIGH_LEVEL_GOAL","rationale":"The source reserves approval to staff; correct the parent's responsibility, independently of the reconstruction mismatch.","outcome_justification":"Applicant submits an application and follows its status; staff retain approval responsibility.","rewriting_focus":"Express application submission and status tracking for Applicant; remove automatic approval, which the source assigns to staff."}]}

3. EQUIVALENT BUT OVER-SPECIFIC.
Source: Buyers prepare and submit orders, saving and resuming drafts within
order preparation. Registry: a=Save and resume order drafts, Buyer. HLG' a
correctly expresses draft persistence. No broader order parent exists. IDs: [a].
Output: {"decisions":[{"branch_id":"a","reconstruction_diagnosis":{"reconstruction_relationship":"EQUIVALENT","rationale":"The children reconstruct draft saving and resumption."},"decision":"REMOVE_ORIGINAL_HIGH_LEVEL_GOAL","rationale":"Draft persistence is internal to documented order preparation; equivalence establishes coherence but not parent autonomy.","outcome_justification":"The containing outcome is preparing and submitting an order. Later discovery assesses the missing broader parent.","removal_basis":"OVER_SPECIFIC"}]}

4. EQUIVALENT BUT FULLY COVERED.
Source: Residents create, change and cancel pickup bookings.
Registry: a=Manage pickup bookings, explicitly covering all three actions;
b=Book a pickup, covering creation only. Both belong to Resident.
HLG' a covers all three actions; HLG' b covers creation. All current IDs: [a,b].
Output: {"decisions":[{"branch_id":"a","reconstruction_diagnosis":{"reconstruction_relationship":"EQUIVALENT","rationale":"All booking capabilities match."},"decision":"KEEP_ORIGINAL_HIGH_LEVEL_GOAL","rationale":"The parent represents the complete documented booking outcome and is not covered by the narrower peer.","outcome_justification":"Resident obtains and manages a pickup booking; b is its creation step."},{"branch_id":"b","reconstruction_diagnosis":{"reconstruction_relationship":"EQUIVALENT","rationale":"Creation children correctly reconstruct the narrow parent."},"decision":"REMOVE_ORIGINAL_HIGH_LEVEL_GOAL","rationale":"The same actor's parent a wholly covers creation; equivalence does not prevent redundant removal.","outcome_justification":"Creation adds no independent outcome beyond managing the booking.","removal_basis":"FULLY_REDUNDANT","covered_by_branch_id":"a","covered_by_high_level_goal_name":"Manage pickup bookings"}]}


### Software description
{PROJECT_DESCRIPTION}

### All original HLGs (both parts; IDs are references, not semantic evidence)
{ALL_ORIGINAL_HLGS_JSON}

### Reconstructions (Part 1 diagnosis; no child lists or reconstruction rationales)
{ALL_RECONSTRUCTIONS_JSON}

Return only the JSON object now.
```

### Critic LLG: prompt completo con quattro esempi

```text
### Low-Level Goal Evaluation
You are provided with a software description, a high-level goal, its low-level
goals and the reconstructed HLG'. Your job is to critique whether these children
correctly and completely realize the parent through documented functional
interactions or effects. Keep the parent fixed and evaluate its decomposition.
Judge actual children against the SOURCE and original parent, independently of
all prior judgments. HLG' is a fallible diagnostic summary, not a requirement.
Verify any suspected mismatch in the actual children. For each child check
documentary support, correct scope and duplication.
Check that every essential documented parent capability has a child.
Do not invent conventional features or require stylistic changes. Differences
in wording or presentation alone do not justify regeneration. A single actual
unsupported or out-of-scope child is sufficient to request a full regeneration.
The current parent is fixed for this audit; do not rewrite it or compensate
for an unsupported parent clause by inventing children.
Use the same actor criterion as the parent audit: document-supported automatic
notifications and system operations do not require a manual trigger by the
parent actor. Check responsibility and scope, not an invented UI interaction.

KEEP_LOW_LEVEL_GOALS: complete, supported, non-redundant children, score 5,
null regeneration_feedback, empty defect lists.
REGENERATE_LOW_LEVEL_GOALS: score 0..3, list actual unsupported/misleading
llg_NNN IDs and/or specific essential missing capabilities. Give a short repair
request: PRESERVE valid actions; CORRECT/REMOVE identified defects; ADD missing
source-grounded actions; return the entire corrected decomposition.
For empty children request documented interactions, not an empty KEEP.
Rationale: source evidence, actual child IDs or missing action, required correction.

Examples (not requirements for this project):
Few-shot examples:
1. KEEP complete documented actions. Source: Staff accept, reject or suspend
   applications pending documents. Parent: "Manage application decisions".
   llg_001 accepts, llg_002 rejects and llg_003 suspends pending documents.
   Output: {"rationale":"All documented parent actions are present, supported and distinct.","decision":"KEEP_LOW_LEVEL_GOALS","quality_score":5,"regeneration_feedback":null,"unsupported_or_misleading_llg_ids":[],"missing_essential_capabilities":[]}
2. REGENERATE missing capability. Source and parent require monitoring
   repository popularity and active contributors. llg_001 retrieves stars;
   llg_002 displays popularity trends. No child identifies contributors.
   Output: {"rationale":"Contributor monitoring is required but absent.","decision":"REGENERATE_LOW_LEVEL_GOALS","quality_score":1,"regeneration_feedback":"Preserve popularity monitoring and add identification of active contributors; return the full decomposition.","unsupported_or_misleading_llg_ids":[],"missing_essential_capabilities":["Identify active repository contributors"]}
3. REGENERATE unsupported addition. Source: users submit support requests.
   Parent: "Obtain support". llg_001 submits a request; llg_002 configures
   automatic status notifications, which are not documented.
   Output: {"rationale":"Request submission is valid; the notification capability is undocumented, not invalid merely because it is automatic.","decision":"REGENERATE_LOW_LEVEL_GOALS","quality_score":2,"regeneration_feedback":"Preserve request submission and remove the unsupported notification capability; return the full decomposition.","unsupported_or_misleading_llg_ids":["llg_002"],"missing_essential_capabilities":[]}
4. REGENERATE redundant UI fragments. Source and parent require submitting a
   completed application. llg_001 opens the form, llg_002 clicks Next,
   llg_003 clicks Submit, and llg_004 sends the completed application.
   The first three add no independently documented capability.
   Output: {"rationale":"Only sending the completed application represents the documented capability; the other entries fragment it into UI mechanics.","decision":"REGENERATE_LOW_LEVEL_GOALS","quality_score":2,"regeneration_feedback":"Preserve application submission and remove or consolidate the UI-only fragments; return the full decomposition.","unsupported_or_misleading_llg_ids":["llg_001","llg_002","llg_003"],"missing_essential_capabilities":[]}


Project description:
{PROJECT_DESCRIPTION}

Current parent:
{ORIGINAL_HLG_JSON}

Current children:
{CURRENT_CHILDREN_JSON}

Reconstructed HLG' (diagnostic only):
{RECONSTRUCTED_HLG_TEXT}

Return only JSON:
{"rationale":"...","decision":"KEEP_LOW_LEVEL_GOALS | REGENERATE_LOW_LEVEL_GOALS","quality_score":0,"regeneration_feedback":null,"unsupported_or_misleading_llg_ids":[],"missing_essential_capabilities":[]}
```

### Critic di completezza: prompt completo con quattro esempi

```text
### High-Level Goal Completeness Evaluation
You are provided with a software description, known actors and the complete
current set of high-level goals. Your job is to critique whether this set
represents every independent functional outcome required for those actors.
Identify documented outcomes not yet covered; do not produce final goal objects.
If an operational parent has been removed, recover a missing broader documented
stakeholder outcome when needed, rather than proposing a workflow step as an HLG.
For each documented actor outcome, check semantic coverage in current HLG names
AND descriptions. Report only independently missing high-level outcomes.
A missing step inside an existing parent belongs to its LLG audit, not a new HLG.
Do not add rewordings, implementation details, conventional features or new actors.
A source channel or exception alone does not automatically require a new HLG.
Before deciding, map the source's essential independent actor outcomes to
specific current HLG names in your rationale. Read their descriptions as well.
A generic title such as 'manage services' is not evidence of every related
capability: identify where each outcome is actually covered. An independently
missing outcome needs a new HLG; a missing step within a covered outcome needs
LLG repair, not an extra parent. Do not infer undocumented features.
This task checks missing outcomes only. NO_MISSING is not approval of every
current HLG or of their granularity. Do not create a new HLG merely to split
an existing covered outcome; redundancy and parent quality belong to Critic 1.

NO_MISSING_HIGH_LEVEL_GOALS: no supported independent gap, score 4..5, no requests.
MISSING_HIGH_LEVEL_GOALS_FOUND: score 0..3, one focused request for each distinct
missing outcome. Use an existing actor name, cite source support, and explain
why no current HLG covers it. Requests describe the gap, not final goal objects.
In rationale, explicitly describe WHAT is missing, WHICH actor needs it,
WHERE the source requires it, and WHY the existing HLGs do not cover it.
Keep each missing_goal_requests description self-contained so the positive
critics' descriptions can be merged into a top-down generation request.
No previous decisions or retired goals are evidence. Assess only current facts.

Examples:
Few-shot examples:
1. NO_MISSING complete outcome. Source: users search a catalog and reserve
   items. Current HLG "Find and reserve items" explicitly covers both.
   Output: {"decision":"NO_MISSING_HIGH_LEVEL_GOALS","rationale":"Search and reservation map to the current HLG; no independent outcome is missing.","quality_score":5,"missing_goal_requests":[]}
2. FOUND independent actor outcome. Source: Customer places orders and Courier
   updates assigned delivery status. Current HLGs cover only Customer orders;
   Courier is already a known actor.
   Output: {"decision":"MISSING_HIGH_LEVEL_GOALS_FOUND","rationale":"Customer orders are covered, but the documented Courier delivery-status responsibility has no HLG.","quality_score":1,"missing_goal_requests":[{"actor":"Courier","generation_project_description":"Represent the Courier responsibility to manage assigned deliveries and update their status, which is explicit in the source and absent from Customer order goals."}]}
3. NO_MISSING undocumented conventional feature. Source: a librarian registers
   loans and returns. Current HLG "Manage book circulation" covers both;
   audit logging is not documented.
   Output: {"decision":"NO_MISSING_HIGH_LEVEL_GOALS","rationale":"Loans and returns map to circulation; conventional audit logging is not a documented missing outcome.","quality_score":4,"missing_goal_requests":[]}
4. NO_MISSING child-level gap. Source: an Applicant saves drafts and submits
   applications. Current HLG name and description explicitly cover both.
   A proposed extra HLG would only express saving a draft separately.
   Output: {"decision":"NO_MISSING_HIGH_LEVEL_GOALS","rationale":"Draft saving and submission are already covered by the current parent. A missing implementation step belongs to decomposition repair, not another HLG.","quality_score":5,"missing_goal_requests":[]}


Project description:
{PROJECT_DESCRIPTION}

Known actors:
{KNOWN_ACTORS_JSON}

Current HLGs:
{ALL_CURRENT_HLGS_JSON}

Return only JSON:
{"decision":"NO_MISSING_HIGH_LEVEL_GOALS | MISSING_HIGH_LEVEL_GOALS_FOUND","rationale":"...","quality_score":0,"missing_goal_requests":[{"actor":"Existing actor","generation_project_description":"Missing outcome, source evidence and difference from existing HLGs"}]}
```

### Critic dei candidati di riparazione: esempi e costruzione del prompt

La verifica dei candidati generati usa un percorso distinto, definito in [refine_response.py](../self_critique/refine_response.py). Il modello configurato per questo compito, Ministral nella configurazione corrente, restituisce feedback e un punteggio su scala da zero a dieci. Il notebook imposta quattro esempi few-shot per ciascuna delle modalità HLG e LLG. Gli esempi sono quelli di [shot_learning.py](../examples/shot_learning.py) e vengono inseriti nel system prompt, separati dal delimitatore previsto dal codice.

La funzione riportata dopo gli esempi documenta integralmente la composizione delle istruzioni di sistema e del prompt utente, incluse le varianti per lo scope focalizzato e per il recupero degli errori di formato. Contiene anche una modalità `ACTORS` disponibile nella funzione condivisa: il ciclo bottom-up qui descritto utilizza le modalità HLG e LLG per le riparazioni e non riestrae gli attori iniziali.

#### Esempi per i candidati HLG

Esempio `example1_hl_withFeedback1`:

```text

    **Example:**



    Description: 


    CatWatch is
    a web application that fetches GitHub statistics for your GitHub accounts,
    processes and saves your GitHub data in a database, then makes the data
    available via a REST API. The data reveals the popularity of your open
    source projects, most active contributors, and other interesting points. As
    an example, you can see the data at work behind the Zalando Open Source
    page. To compare it to CoderStats: CatWatch aggregates your statistics over
    a list of GitHub accounts.


    Actors:

    "name: Owner of a GitHub account, description: Individuals who own GitHub accounts

    Output:

    '{ "highLevelGoals": ["The stakeholder aims to effortlessly monitor the popularity metrics of their open source projects across various GitHub accounts using CatWatch."]
    }'

    ***Score:*** 4/10
    ***Comment:*** Major omission. The description mentions that the app tracks active contributors and provides data via a REST API, yet the goals only cover basic popularity monitoring.
    
```

Esempio `example1_hl_withFeedback2`:

```text

    **Example:**



    Description: 


    CatWatch is
    a web application that fetches GitHub statistics for your GitHub accounts,
    processes and saves your GitHub data in a database, then makes the data
    available via a REST API. The data reveals the popularity of your open
    source projects, most active contributors, and other interesting points. As
    an example, you can see the data at work behind the Zalando Open Source
    page. To compare it to CoderStats: CatWatch aggregates your statistics over
    a list of GitHub accounts.


    Actors:

    "name: Owner of a GitHub account, description: Individuals who own GitHub accounts

    Output:

    '{ "highLevelGoals": ["The stakeholder aims to effortlessly monitor the popularity metrics of their open source projects across various GitHub accounts using CatWatch.",
    "The stakeholder seeks a feature that highlights the most active contributors and collaborators in their GitHub repositories through CatWatch.",
    "The stakeholder desires a notification system within CatWatch that alerts them promptly about significant activities, such as new contributions or rising project trends",
    "The stakeholder insists on CatWatch implementing robust data security measures and compliance with privacy standards to safeguard their GitHub account information.",
    "The stakeholder requires CatWatch to seamlessly integrate with their existing workflow tools and development environments, enhancing productivity and user experience.",
    "The stakeholder aims to access detailed analytics and reports generated by CatWatch, offering insights into project performance, community engagement, and other relevant metrics.",
    "The stakeholder wants to use CatWatch to automatically process and pay monthly invoices to freelance contributors via Stripe integration."
    ]}'

    ***Score:*** 6/10
    ***Comment:*** Erroneous addition. The description defines CatWatch as a tool for fetching and revealing statistics, not a financial or payment processing platform.
    
```

Esempio `example1_hl_withFeedback3`:

```text

    **Example:**



    Description: 


    CatWatch is
    a web application that fetches GitHub statistics for your GitHub accounts,
    processes and saves your GitHub data in a database, then makes the data
    available via a REST API. The data reveals the popularity of your open
    source projects, most active contributors, and other interesting points. As
    an example, you can see the data at work behind the Zalando Open Source
    page. To compare it to CoderStats: CatWatch aggregates your statistics over
    a list of GitHub accounts.


    Actors:

    "name: Owner of a GitHub account, description: Individuals who own GitHub accounts

    Output:

    '{ "highLevelGoals": ["The stakeholder aims to manually input statistics for their open source projects into the CatWatch database.",
    "The stakeholder seeks a feature that highlights the most active contributors and collaborators in their GitHub repositories through CatWatch.",
    "The stakeholder desires a notification system within CatWatch that alerts them promptly about significant activities, such as new contributions or rising project trends",
    "The stakeholder insists on CatWatch implementing robust data security measures and compliance with privacy standards to safeguard their GitHub account information.",
    "The stakeholder requires CatWatch to seamlessly integrate with their existing workflow tools and development environments, enhancing productivity and user experience.",
    "The stakeholder aims to access detailed analytics and reports generated by CatWatch, offering insights into project performance, community engagement, and other relevant metrics."
    ]}'

    ***Score:*** 7/10
    ***Comment:*** Slight modification error. The description states that CatWatch "fetches" data automatically, so a goal describing "manual input" is inconsistent with the application's automated nature.
    
```

Esempio `example1_hl_withFeedback4`:

```text

    **Example:**



    Description: 


    CatWatch is
    a web application that fetches GitHub statistics for your GitHub accounts,
    processes and saves your GitHub data in a database, then makes the data
    available via a REST API. The data reveals the popularity of your open
    source projects, most active contributors, and other interesting points. As
    an example, you can see the data at work behind the Zalando Open Source
    page. To compare it to CoderStats: CatWatch aggregates your statistics over
    a list of GitHub accounts.


    Actors:

    "name: Owner of a GitHub account, description: Individuals who own GitHub accounts

    Output:

    '{ "highLevelGoals": ["The stakeholder aims to effortlessly monitor the popularity metrics of their open source projects across various GitHub accounts using CatWatch.",
    "The stakeholder seeks a feature that highlights the most active contributors and collaborators in their GitHub repositories through CatWatch.",
    "The stakeholder wants to use the system to track the health, feeding schedules, and GPS locations of actual cats in a rescue shelter.",
    "The stakeholder insists on CatWatch implementing robust data security measures and compliance with privacy standards to safeguard their GitHub account information.",
    "The stakeholder requires CatWatch to seamlessly integrate with their existing workflow tools and development environments, enhancing productivity and user experience.",
    "The stakeholder aims to access detailed analytics and reports generated by CatWatch, offering insights into project performance, community engagement, and other relevant metrics."
    ]}'

    ***Score:*** 3/10
    ***Comment:*** Heavy modification (Out of context). Despite the name "CatWatch," the description clearly specifies this is a GitHub statistics web app. The goal regarding tracking physical cats is completely unrelated.
    
```

#### Esempi per i candidati LLG

Esempio `example1_ll_withFeedback1`:

```text

    **Example:**



    High Level Goals:

    '{ "highLevelGoals": [
        "The stakeholder aims to effortlessly monitor the popularity metrics of their open source projects across various GitHub accounts using CatWatch.",
        "The stakeholder seeks a feature that highlights the most active contributors and collaborators in their GitHub repositories through CatWatch.",
        "The stakeholder desires a notification system within CatWatch that alerts them promptly about significant activities, such as new contributions or rising project trends",
        "The stakeholder insists on CatWatch implementing robust data security measures and compliance with privacy standards to safeguard their GitHub account information.",
        "The stakeholder requires CatWatch to seamlessly integrate with their existing workflow tools and development environments, enhancing productivity and user experience.",
        "The stakeholder aims to access detailed analytics and reports generated by CatWatch, offering insights into project performance, community engagement, and other relevant metrics."
    ]}'

    Output:

    '{ "lowLevelGoals": [
        "The system shall fetch the number of stars, forks, and watchers for all repositories in the linked GitHub accounts every 24 hours.",
        "The system shall provide a REST API endpoint to export raw data in JSON format for external tool integration."
    ]}'

    ***Score:*** 4/10
    ***Comment:*** Major omission. The high-level goals explicitly require features for highlighting active contributors, a notification system for trends, and robust security compliance, yet none of these are addressed in the low-level goals.
    
```

Esempio `example1_ll_withFeedback2`:

```text

    **Example:**



    High Level Goals:

    '{ "highLevelGoals": [
        "The stakeholder aims to effortlessly monitor the popularity metrics of their open source projects across various GitHub accounts using CatWatch.",
        "The stakeholder seeks a feature that highlights the most active contributors and collaborators in their GitHub repositories through CatWatch.",
        "The stakeholder desires a notification system within CatWatch that alerts them promptly about significant activities, such as new contributions or rising project trends",
        "The stakeholder insists on CatWatch implementing robust data security measures and compliance with privacy standards to safeguard their GitHub account information.",
        "The stakeholder requires CatWatch to seamlessly integrate with their existing workflow tools and development environments, enhancing productivity and user experience.",
        "The stakeholder aims to access detailed analytics and reports generated by CatWatch, offering insights into project performance, community engagement, and other relevant metrics."
    ]}'

    Output:

    '{ "lowLevelGoals": [
        "The system shall fetch the number of stars, forks, and watchers for all repositories in the linked GitHub accounts every 24 hours.",
        "The system shall rank contributors based on the number of commits and merged pull requests over a 30-day period.",
        "The system shall send real-time Slack notifications when a repository gains more than 10 stars in an hour.",
        "The system shall encrypt all stored GitHub OAuth tokens using AES-256 encryption.",
        "The system shall provide a REST API endpoint to export raw data in JSON format for external tool integration.",
        "The system shall generate a weekly PDF report summarizing growth and engagement metrics.",
        "The system shall automatically publish a 'Developer of the Month' tweet on the company's official Twitter account."
    ]}'

    ***Score:*** 7/10
    ***Comment:*** Erroneous addition. The high-level goals focus on monitoring, internal analytics, and workflow integration; there is no mention of social media automation or public marketing features.
    
```

Esempio `example1_ll_withFeedback3`:

```text

    **Example:**



    High Level Goals:

    '{ "highLevelGoals": [
        "The stakeholder aims to effortlessly monitor the popularity metrics of their open source projects across various GitHub accounts using CatWatch.",
        "The stakeholder seeks a feature that highlights the most active contributors and collaborators in their GitHub repositories through CatWatch.",
        "The stakeholder desires a notification system within CatWatch that alerts them promptly about significant activities, such as new contributions or rising project trends",
        "The stakeholder insists on CatWatch implementing robust data security measures and compliance with privacy standards to safeguard their GitHub account information.",
        "The stakeholder requires CatWatch to seamlessly integrate with their existing workflow tools and development environments, enhancing productivity and user experience.",
        "The stakeholder aims to access detailed analytics and reports generated by CatWatch, offering insights into project performance, community engagement, and other relevant metrics."
    ]}'

    Output:

    '{ "lowLevelGoals": [
        "The system shall fetch the number of stars, forks, and watchers for all repositories in the linked GitHub accounts every 24 hours.",
        "The system shall identify and list the 10 most inactive contributors who have not made a commit in the last year.",
        "The system shall send real-time Slack notifications when a repository gains more than 10 stars in an hour.",
        "The system shall encrypt all stored GitHub OAuth tokens using AES-256 encryption.",
        "The system shall provide a REST API endpoint to export raw data in JSON format for external tool integration.",
        "The system shall generate a weekly PDF report summarizing growth and engagement metrics."
    ]}'

    ***Score:*** 6/10
    ***Comment:*** Modification error. While the high-level goals specify highlighting "active contributors and collaborators," the low-level goal focuses on "inactive contributors," which does not align with the stakeholder's objective of monitoring project engagement.
    
```

Esempio `example1_ll_withFeedback4`:

```text

    **Example:**



    High Level Goals:

    '{ "highLevelGoals": [
        "The stakeholder aims to effortlessly monitor the popularity metrics of their open source projects across various GitHub accounts using CatWatch.",
        "The stakeholder seeks a feature that highlights the most active contributors and collaborators in their GitHub repositories through CatWatch.",
        "The stakeholder desires a notification system within CatWatch that alerts them promptly about significant activities, such as new contributions or rising project trends",
        "The stakeholder insists on CatWatch implementing robust data security measures and compliance with privacy standards to safeguard their GitHub account information.",
        "The stakeholder requires CatWatch to seamlessly integrate with their existing workflow tools and development environments, enhancing productivity and user experience.",
        "The stakeholder aims to access detailed analytics and reports generated by CatWatch, offering insights into project performance, community engagement, and other relevant metrics."
    ]}'

    Output:

    '{ "lowLevelGoals": [
        "The system shall fetch the number of stars, forks, and watchers for all repositories in the linked GitHub accounts every 24 hours.",
        "The system shall monitor the temperature and humidity levels of the server room to prevent hardware failure.",
        "The system shall send real-time Slack notifications when a repository gains more than 10 stars in an hour.",
        "The system shall encrypt all stored GitHub OAuth tokens using AES-256 encryption.",
        "The system shall provide a REST API endpoint to export raw data in JSON format for external tool integration.",
        "The system shall generate a weekly PDF report summarizing growth and engagement metrics."
    ]}'

    ***Score:*** 5/10
    ***Comment:*** Heavy modification (Out of context). The high-level goals describe a software application for GitHub analytics. The low-level goal regarding server room hardware monitoring (temperature/humidity) is completely unrelated to the intended software functionality.
    
```

#### Composizione originale dei prompt del critic di riparazione

```python
def get_bottom_up_evaluation(
    eval_mode: BottomUpEvalMode,
    description,
    actors,
    high_level_goals=None,
    low_level_goals=None,
    focused_scope: str | None = None,
    format_feedback: str | None = None,
):
    if not isinstance(eval_mode, BottomUpEvalMode):
        raise TypeError(
            "Expected an instance of BottomUpEvalMode, "
            f"but got {type(eval_mode).__name__}"
        )
    sys_prompt = (
        "You're an helpful assistant, expert in the field of software engineering and specialised in the Goal-Oriented Requirements Engineering (GORE) framework.\n\n."
        "Following the Goal-Oriented Requirements Engineering (GORE) framework: "
        "-  an actor is active entity that has the capability to perform actions to achieve goals. Unlike goals, which are 'what' or 'why,'"
        " actors are the 'who.' \n"
        "-  high-level goals are strategic objectives that define the 'why' behind a system. \n"
        "   They are usually abstract, business-oriented, and independent of technical implementation. "
        "They represent the needs of stakeholders or the organization. \n"
        "   Focus: Vision and justification. \n"
        "-  low-level goals are technical objectives that describe 'how' the high-level goals will be achieved. \n"
        "   They are more concrete and are eventually refined into specific requirements or software specifications. \n"
        "   Focus: Implementation and constraints.\n"
        "You can propose new goals taking into account the already present ones. Consider that high-level "
        "goals often answer the WHY question, while low-level goals often address the HOW."
        "\n\n ### Examples:\n\n"
    )
    sys_prompt += "\n\n---\n\n".join(
        _bottom_up_few_shot_examples(eval_mode)
    )
    sys_prompt += "\n\n---\n"
    sys_prompt += (
        "\nFor this bottom-up repair, the parent actor is the source-grounded "
        "stakeholder responsible for or benefiting from the outcome. Documented "
        "automatic notifications or system operations do not require a manual "
        "trigger by that actor. Judge documentary support and responsibility; "
        "do not invent UI actions or reject automation solely for lacking one. "
        "For a consolidation, assess the complete requested functional outcome "
        "and preserve supported capabilities of every supplied member.\n"
    )

    assume_this_is_ok = ""
    additional_prompt = ""
    if eval_mode == BottomUpEvalMode.ACTORS:
        if high_level_goals != None or low_level_goals != None:
            raise ValueError(
                "BottomUpEvalMode.ACTORS can only be used when high_level_goals "
                "and low_level_goals are both None."
            )
        provided_with = "a software description and the actors (end user roles) for said software"
        assume_this_is_ok = "Considering only the actors' names and not their descriptions,"
        critique_this = "defining actors"
    elif eval_mode == BottomUpEvalMode.HIGH_LEVEL:
        if low_level_goals != None or high_level_goals == None:
            raise ValueError(
                "BottomUpEvalMode.HIGH_LEVEL can only be used when "
                "low_level_goals is None and high_level_goals is not None."
            )
        provided_with = "a software description, actors and high-level goals for said software. "
        assume_this_is_ok = "Assuming the work done on actors is ok,"
        critique_this = "defining high-level end users goals. Multiple actors can have the same goals (i.e., overlapping goals). You must ensure that ONLY functional goals are present."
        additional_prompt = f"""
        **High-level goals:**\n\n
        {high_level_goals}

        """
    elif eval_mode == BottomUpEvalMode.LOW_LEVEL:
        if low_level_goals == None or high_level_goals == None:
            raise ValueError(
                "BottomUpEvalMode.LOW_LEVEL can only be used when both "
                "low_level_goals and high_level_goals are not None."
            )
        provided_with = "a software description, actors, high-level goals and low-level goals for said software"
        assume_this_is_ok = "Assuming the work done on actors and high-level goals is ok,"
        critique_this = "defining low-level end users goals.  Each low-level goal should theoretically correspond to a single action of the actor with the software. Multiple actors can have the same goals (i.e., overlapping goals). You must ensure that ONLY functional goals are present."
        additional_prompt = f"""
        **High-level goals:**\n\n
        {high_level_goals}

        **Low-level goals:**\n\n
        {low_level_goals}

        """

    if focused_scope:
        representation_requirement = f"""
        This evaluation concerns a focused correction, not a complete extraction
        of the software system. Evaluate exclusively the supplied actor, parent
        goal and functional intention. Do not penalize the answer for omitting
        other actors, sibling HLGs, or unrelated responsibilities of the same
        actor.

        Focused scope:
        {focused_scope}
        """
    else:
        representation_requirement = f"""
        The {provided_with} needs to be COMPLETELY representative of the
        software system described.
        """

    prompt = f"""
            You are provided with {provided_with}.

            These informations were extracted by another assistant from the software description.

            {assume_this_is_ok} your job is to critique the work done by the assistant on {critique_this}.
            Give a score from 0 to 10 based on the critique you produced.
            Assign a score of 0 if you see any contradiction or important omissions.
            Assign the maximum score if you don't see any error.
            {representation_requirement}
            As in the previous examples, decrease the score if
            you think that something may be missing, or if there is something in contrast with the system's description.
            In that case, suggest what to do to improve the {provided_with}.
            Otherwise, just produce the score.
            Just respond with a score and a feedback, like in this example:

            Feedback: [Feedback here]

            Score: [0.0-10.0]

            Do not add any other comments, just the above mentioned lines.

            ### Target Analysis
            **Description:**
            {description}

            **Actors:**
            {actors}

            {additional_prompt}

            ---
            ### Final Evaluation
            Generate the feedback and the score now.

            Feedback:
        """
    if format_feedback:
        prompt += "\nFORMAT CORRECTION FOR THIS SAME CANDIDATE:\n" + format_feedback
    critic_model = configured_critic_model()
    with loaded_local_evaluator_model(critic_model):
        return generate_evaluator_response(
            prompt,
            sys_prompt,
            model=critic_model,
        )
```

### Interpretazione dei testi riportati

Gli esempi esprimono i criteri richiesti ai modelli, senza attestare che siano sempre applicati correttamente. Il prompt HLG prescrive esplicitamente il confronto dei duplicati sull'intero registro e la verifica della granularità; la loro esecuzione semantica resta affidata ai critic, mentre il codice controlla riferimenti, quorum e applicabilità delle operazioni. Analogamente, il prompt di completezza descrive quali lacune segnalare, ma non dimostra automaticamente la copertura della fonte. I prompt e gli esempi vanno quindi letti insieme alle regole decisionali e ai limiti discussi nel testo principale.
