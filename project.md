# AstroFit — Design dell'API

AstroFit è una libreria Julia per la **creazione e il fitting di modelli parametrici**.
Il *motore* esiste già ed è testato: vincoli-come-tipi, aggiornamento immutabile via
[`Accessors`](https://github.com/JuliaObjects/Accessors.jl), ricostruzione del modello da
un vettore piatto type-stable, zero-alloc, con autodiff (`ForwardDiff`) che fluisce anche
attraverso i vincoli funzionali.

Questo documento descrive il **layer di authoring** sopra al motore: come l'utente costruisce
un modello nominato, vi attacca vincoli complessi, e definisce *prefab* (modelli con vincoli
fisici già corretti) che restano componibili. Il design dell'interfaccia di fit
(scelta dell'optimizer, loss, formato dati) è **rimandato** a un documento dedicato.

---

## Sezione 1 — L'API di alto livello

### Il modello mentale

C'è **un oggetto centrale** e tutto vi si aggancia: il modello. I *valori* dei parametri
(es. `amplitude = 2.0`) vivono dentro il modello; i *vincoli* sono metadati associati.
Di conseguenza **non esiste** una funzione `render(x, model, constraints)`: per valutare un
modello lo si chiama direttamente. La valutazione è *pointwise* — `cm(x)` valuta in un
punto, sulla griglia si usa il broadcast `cm.(xs)`. Valutare e fittare restano operazioni
distinte.

### I tre mattoni

| Concetto | Cos'è | Esempio |
|---|---|---|
| **Model** | Un modello *nudo*: pura funzione parametrica, senza vincoli né nomi. | `Gaussian1D(amplitude=2.0, sigma=1.0)` |
| **CompiledModel** | Un Model + i suoi vincoli + la mappa dei nomi, "compilato". È l'oggetto su cui si lavora: chiamabile, fittabile, aggiornabile, componibile. | risultato di `@model` / `@constrain` |
| **Prefab** | Una funzione che restituisce un CompiledModel con i vincoli fisici già applicati. | `EmissionLine(center=6563)` |

> **Nota terminologica.** Il `CompiledModel` *è* il tipo che nel codice attuale si chiama
> `ParametricModel`: un solo concetto, non due tipi paralleli.

### Le due macro

**`@model`** — costruisce topologia e nomi. La regola è una sola: **ogni foglia
dell'espressione di composizione è un componente nominato, e il suo nome è il suo simbolo.**
Da qui due forme, mescolabili:

- *a blocco*: i binding a sinistra dell'`=` definiscono quei simboli localmente, e l'ultima
  espressione descrive come si combinano;
- *inline*: `@model model1 + model2` prende i simboli dallo scope circostante — i nomi dei
  componenti diventano `model1`, `model2`.

Le foglie restano pure (nessun campo `name`): i nomi vivono nel CompiledModel risultante.
Conseguenze della regola "nomi obbligatori": una foglia anonima inline
(`model1 + Gaussian1D(...)`) è un **errore** (legala prima a un nome); lo stesso simbolo due
volte è una **collisione** fra fratelli; una foglia che è già un `CompiledModel` (un prefab)
ne fa **assorbire la spec**, namespaced sotto il nome del binding.

**`@constrain model begin … end`** — attacca i vincoli e **restituisce direttamente il
CompiledModel** (risolve i nomi e compila in un colpo solo). Funziona sia su un Model nudo
sia su un CompiledModel già pronto (un prefab): in quel caso fa **merge per nome**, e i
vincoli dell'utente **sovrascrivono** quelli di fabbrica sullo stesso parametro (l'ultimo
vince), lasciando intatti gli altri.

### Gli operatori di composizione

I componenti si combinano con normali operatori Julia; ognuno produce un nodo dedicato
nell'albero del modello:

| Operatore | Nodo | Semantica | Dim |
|---|---|---|---|
| `a + b`   | `Sum`        | `a(x) + b(x)`            | stessa N |
| `a * b`   | `Product`    | `a(x) * b(x)`            | stessa N |
| `a - b`   | `Difference` | `a(x) - b(x)`            | stessa N |
| `a / b`   | `Quotient`   | `a(x) / b(x)`            | stessa N |
| `a ∘ b`   | `Pipe`       | `a(b(x))` — `a` è 1D     | `b` qualsiasi |
| `a \|> b` | `Pipe`       | `b(a(x))` — `b` è 1D     | `a` qualsiasi |

`|>` è `∘` con gli argomenti invertiti: la lettura "pipeline" (`dato |> trasforma`) per chi
preferisce l'ordine di flusso a quello matematico. Attenzione al ribaltamento di dimensione:
in `a |> b` è **b** il modello 1D esterno.

**Scalari.** L'algebra è modello⊗modello: niente scalari nudi (`2 * model`, `model + 1`,
`-model`). Per una costante usa un `Const1D` nominato — così resta fittabile, vincolabile e
indirizzabile come ogni altro componente, e la regola "nomi obbligatori" non ha eccezioni.
(Decisione rivedibile se l'uso di `2 * Gaussian1D(...)` dovesse diventare frequente.)

**CompiledModel fuori da `@model`.** Per la stessa ragione l'algebra non è definita sui
`CompiledModel` fuori da `@model`: `Ha + Hb` nudo dovrebbe scartare le spec in silenzio.
Entrambi i casi danno un **errore informativo** — «componi dentro `@model`, così i vincoli
viaggiano»; «usa un `Const1D` nominato» — mai una perdita silenziosa.

### I quattro keyword dei vincoli

Mappano 1:1 sui quattro tipi-vincolo del motore (`Free`/`Fixed`/`Bounded`/`Tied`):

| Keyword | Significato | Forme |
|---|---|---|
| `@fix`   | Fissa un parametro, escluso dal fit | `@fix p = v` (a un valore) · `@fix p` (al valore corrente) |
| `@bound` | Limita un parametro all'intervallo | `@bound p in (lo, hi)` (usa `Inf` per un lato solo) |
| `@tie`   | Parametro *dipendente*, calcolato da altri | `@tie p = expr(altri.param)` (i master sono auto-rilevati) |
| `@free`  | Rilascia un parametro | `@free p` (tipicamente per disfare un vincolo di fabbrica) |

`@tie` accetta un'espressione qualsiasi sugli altri parametri. Vincolo del motore: un master
non può essere a sua volta legato (niente catene di `tie`) — l'API segnala errore.

### Indirizzamento dei parametri

I componenti di un modello composito si indirizzano **per nome esplicito**: `narrow.amplitude`,
`broad.sigma`. Un modello singolo non richiede nome (`amplitude` basta). Componendo prefab i
nomi diventano gerarchici (`Ha.line.sigma`), il che li disambigua automaticamente.

### Ciclo di vita (immutabile)

```
@model  ─►  CompiledModel  ──@constrain──►  CompiledModel  ──fit──►  CompiledModel
                  │                                │
               cm.(xs)  valuta                 @set  aggiorna un valore (ritorna nuovo)
                                                withparams(cm, p)  ricostruisce dal
                                                vettore piatto dei liberi (idem)
```

Tutto è immutabile, in linea con `Accessors`: `@set cm.narrow.amplitude = 3.0` ritorna un
*nuovo* CompiledModel; `fit(cm, data)` ritorna un *nuovo* CompiledModel coi valori fittati.

**Invariante (albero sempre risolto).** Il modello dentro un CompiledModel è *sempre*
tie-risolto: ogni operazione che cambia stato (`@constrain`, `@set`, `withparams`, `fit`)
ri-risolve i tie prima di restituire. Così `cm(x)` è pura valutazione: nessun lavoro sui
vincoli nel path caldo.

### Componibilità dei prefab

Mettere un prefab dentro `@model` ne fa **viaggiare i vincoli**, namespaced dal nome del
binding. `Ha = EmissionLine()` porta il suo `line.sigma > 0` come `Ha.line.sigma > 0` nel
modello combinato; due prefab dello stesso tipo non collidono perché i prefissi differiscono
(`Ha.` vs `Hb.`). È ciò che rende i prefab realmente riusabili nel caso astronomico comune:
continuo + N righe, ognuna con la sua fisica.

---

## Sezione 2 — Esempi di codice dell'API finale

### 2.1 — Modello composito, vincoli, valutazione

```julia
# Costruzione: nomi dai binding, topologia dall'ultima espressione
model = @model begin
    narrow = Gaussian1D(amplitude=2.0, sigma=1.0)
    broad  = Gaussian1D(amplitude=0.5, sigma=8.0)
    narrow + broad
end

# Vincoli: ritorna direttamente il CompiledModel
cm = @constrain model begin
    @fix   narrow.amplitude = 1.0
    @bound narrow.mean      in (-1, 1)
    @bound broad.sigma      in (0, Inf)
    @tie   broad.mean       = narrow.mean          # tie identità
    @tie   broad.amplitude  = narrow.amplitude / 3 # tie funzionale
end

y  = cm(2.5)    # valuta in un punto (l'albero è già tie-risolto)
ys = cm.(xs)    # sulla griglia: broadcast esplicito
```

### 2.2 — Aggiornamento e fit (immutabili)

```julia
val = cm.narrow.amplitude               # lettura per nome
cm2 = @set cm.narrow.amplitude = 3.0    # nuovo CompiledModel, spec invariata
cm3 = fit(cm, data)                     # nuovo CompiledModel coi valori fittati
cm4 = withparams(cm, p)                 # nuovo CompiledModel dal vettore piatto dei liberi

# il fit loop interno è esattamente questo:
loss(p) = sum(abs2, withparams(cm, p).(xs) .- ys)

# Accesso esplicito quando serve (o quando un nome collide con un campo interno):
amp = cm[:narrow].amplitude
```

### 2.3 — Definire un prefab

Un prefab è semplicemente una funzione che restituisce un CompiledModel con i vincoli
fisici già dentro:

```julia
function EmissionLine(; center, flux=1.0)
    m = @model begin
        line = Gaussian1D(mean=center, amplitude=flux)
        line
    end
    @constrain m begin
        @bound line.sigma in (0, Inf)    # fisica: larghezza positiva
    end
end
```

### 2.4 — Comporre prefab (i vincoli viaggiano namespaced)

```julia
spectrum = @model begin
    cont = Linear1D(slope=0.0, intercept=1.0)   # modello nudo
    Ha   = EmissionLine(center=6563.0)           # prefab → CompiledModel
    Hb   = EmissionLine(center=4861.0)           # prefab → CompiledModel
    cont + Ha + Hb
end
# spectrum porta automaticamente:  Ha.line.sigma > 0  e  Hb.line.sigma > 0
```

### 2.5 — Adattare un modello con prefab (override + tie + free)

```julia
cm = @constrain spectrum begin
    @bound Ha.line.sigma in (1, 5)        # restringe il bound di fabbrica (override)
    @tie   Hb.line.sigma = Ha.line.sigma  # lega le due larghezze
    @free  Ha.line.amplitude              # rilascia un eventuale vincolo di fabbrica
end

result = fit(cm, data)
```

### 2.6 — Forme di `@model` e operatori

`@model` accetta sia un blocco sia un'espressione inline su modelli già nello scope; le due
forme si mescolano, e il nome di ogni componente è il simbolo della foglia.

```julia
g1 = Gaussian1D(amplitude=2.0, mean=0.0, sigma=1.0)
g2 = Gaussian1D(amplitude=1.0, mean=5.0, sigma=2.0)

# inline: i nomi dei componenti sono `g1` e `g2`
two_peaks = @model g1 + g2
two_peaks.g1.sigma                # 1.0 — indirizzati per nome di variabile

# mista: g1/g2 dallo scope, `base` definito localmente
with_base = @model begin
    base = Const1D(value=0.5)
    g1 + g2 + base
end
```

Gli operatori coprono somma / prodotto / differenza / quoziente e la composizione:

```julia
spectrum = @model begin
    line  = Gaussian1D(amplitude=1.0, mean=0.0, sigma=1.0)
    absor = Gaussian1D(amplitude=0.3, mean=0.0, sigma=3.0)
    cont  = Linear1D(slope=0.0, intercept=1.0)
    cont + line - absor             # Difference annidata in una Sum
end

# composizione: l'uscita di `inner` entra in `outer` (outer dev'essere 1D)
inner  = Linear1D(slope=2.0, intercept=1.0)
outer  = Gaussian1D(amplitude=1.0, mean=0.0, sigma=1.0)
shaped = @model inner |> outer      # outer(inner(x)),  ≡  outer ∘ inner
```

---

## Sezione 3 — Come potremmo implementarlo

> Linee guida, non specifica definitiva: la progettazione era sull'interfaccia. Il punto di
> partenza è che **il motore esiste già** e va riusato, non riscritto.

### 3.1 — Riuso del motore esistente

Restano invariati nella sostanza: i tipi-vincolo, l'aggiornamento via `Accessors.set`, la
ricostruzione del modello da un vettore piatto (l'attuale `pm(p)`, che diventa la funzione
nominata `withparams` — v. §3.4), le utility
(`free_lenses`, `gather`/`scatter`, `bounds_vectors`, `nfree`, `freevals`, `paramvector`),
e le garanzie di hot-path (type-stable, `@allocated == 0`, autodiff attraverso i vincoli
funzionali).

### 3.2 — Rinominazioni e nuovi nodi operatore

Rinominazioni:
- `Coupled → Tied` (per allinearsi a `@tie`) — `src/constraints.jl`, usi, test, export.
- `ParametricModel → CompiledModel` — `src/params.jl`, `src/AstroFit.jl`, test, docstring.
- la callable di ricostruzione `pm(p)` → funzione nominata `withparams(pm, p)` (§3.4):
  libera la sintassi di chiamata per la valutazione in `x` ed elimina l'ambiguità di
  `cm(vettore)` (vettore di parametri o griglia di punti?).

Nuovi nodi (si **aggiungono**, non toccano `Sum`/`Product`/`Pipe` né i loro test):
- in `src/compound.jl`: `Difference` (`a - b`) e `Quotient` (`a / b`) — pointwise, stessa `N`,
  con eval per `{1}` e `{2}` sul modello di `Sum`/`Product`;
- in `src/abstract.jl`: i metodi `Base.:-` / `Base.:/` per costruirli, e un metodo `Base.:|>`
  che costruisce un `Pipe` con argomenti invertiti rispetto a `∘` (`a |> b` ≡ `b ∘ a`; in
  `a |> b` il modello 1D esterno è `b`).

### 3.3 — Anatomia del CompiledModel

```julia
struct CompiledModel{M, S, R}
    model::M    # albero nudo, valori dentro — sempre tie-risolto (invariante I1)
    spec::S     # collezione ordinata di coppie optic_target => Constraint (Tuple, non Dict)
    names::R    # registry gerarchica nome → optic (NamedTuple annidata)
end
```

**I valori vivono nell'albero, i vincoli sono marker** (estensione dell'ADR 1). I
tipi-vincolo non portano valori di parametri: `Free` e `Fixed` sono marker puri,
`Bounded(lo, hi)` porta solo i limiti, `Tied(masters, f)` gli optic dei master e la
funzione. `@fix p = v` scrive `v` *nell'albero* e marca `Fixed`: nessuna doppia copia da
tenere sincronizzata. I parametri non menzionati nella spec sono `Free` per default, e la
valutazione non consulta mai la spec.

**Invariante I1 (albero sempre risolto).** I valori dei parametri `Tied` dentro `model`
sono sempre coerenti coi master. Tutti i costruttori passano da un unico cancello interno:

```julia
_compiled(model, spec, names) = CompiledModel(resolve(model, spec), spec, names)
```

Lo mantengono `@constrain`, `@set`, `withparams` e `fit`; nessun altro punto del codice
crea un `CompiledModel` direttamente.

**La registry `names`.** NamedTuple annidata che rispecchia la gerarchia dei componenti:
chiave = nome del componente; valore = l'optic verso quel sottoalbero più, per i prefab,
la sotto-registry dei loro componenti interni. NamedTuple e non Dict, così `getproperty`
con simbolo letterale è type-stable per constant propagation. È il meccanismo unico che
alimenta tre funzionalità:
1. la risoluzione dei nomi in `@constrain` (`narrow.amplitude` → optic),
2. l'accesso/aggiornamento `cm.narrow.amplitude` e `@set`,
3. il merge/override per nome.

`compile(model, spec)` resta come API di basso livello; guadagna una variante che accetta
anche la registry.

### 3.4 — Semantica di valutazione e risoluzione eager

Il "rendering" è **pura valutazione ricorsiva di un albero già risolto**: grazie a I1 non
c'è alcuna logica di vincoli nel path di valutazione.

```julia
(cm::CompiledModel)(x) = getfield(cm, :model)(x)   # tutto qui
```

La valutazione è pointwise (`cm(x)` su un punto, `cm.(xs)` sulla griglia). I nodi compound
sono binari e immutabili, parametrici nei tipi dei figli: la forma dell'albero sta nel
tipo, la ricorsione viene inlined dal compilatore, zero alloc.

**`resolve(model, spec)`** — la funzione che stabilisce I1: per ogni voce
`target => Tied(masters, f)` esegue

```julia
model = set(model, target, f(get(model, m1), get(model, m2), ...))
```

Passata singola e ordine irrilevante: i master non possono essere a loro volta `Tied`
(niente catene, validato al merge in §3.6), quindi non esistono dipendenze fra tie.

**`withparams(cm, p)`** — la ricostruzione dal vettore piatto (l'ex callable `pm(p)`):

```julia
withparams(cm, p) = _compiled(scatter(cm.model, free_lenses(cm), p), cm.spec, cm.names)
```

`p` contiene i valori di `Free` **e** `Bounded` (i bounded si fittano, con box
constraints), nell'ordine delle `free_lenses` del motore. Essendo tutto immutabile e
isbits-friendly resta zero-alloc nel fit loop, e l'autodiff fluisce: i `Dual` entrano da
`p`, attraversano lo scatter e le funzioni dei tie dentro `resolve`.

```julia
# il fit loop interno è esattamente questo:
loss(p) = sum(abs2, withparams(cm, p).(xs) .- ys)
```

La callable `pm(p)` **viene rimossa**: la chiamata su un CompiledModel significa solo
«valuta in x» — altrimenti `cm(vettore)` sarebbe ambiguo (vettore di parametri o griglia?).

### 3.5 — La macro `@model`

Una sola strada copre la forma a blocco e quella inline: emettere gli eventuali
assegnamenti del blocco e poi delegare a un **builder a closure**. La macro non lascia
che l'espressione utente costruisca l'albero da sola: dall'AST estrae i nomi delle foglie,
l'optic di ogni foglia e una closure che ricostruisce l'albero, e a runtime chiama il
builder. La forma inline è la stessa strada con zero assegnamenti.

```julia
# @model begin narrow = ...; broad = ...; narrow + broad end   ⇒   (concettualmente)
_build_model((:narrow, :broad), (optic₁, optic₂), (narrow, broad)) do a, b
    a + b
end
```

Il builder, per ogni foglia:

- **Model nudo** → entra com'è; registry: nome → optic. Il nome è il simbolo della foglia
  (nel blocco definito localmente, inline preso dallo scope circostante). Foglia anonima
  → errore all'espansione; simbolo ripetuto → collisione fra fratelli; nome riservato
  (`model`/`spec`/`names`) → errore.
- **CompiledModel (un prefab)** → lo **scarta** al suo albero nudo per la composizione;
  la sua spec viene **ri-rootata** sotto il prefisso del nome e la sua registry diventa
  la sotto-registry di quel nome.

Poi costruisce l'albero chiamando la closure sui valori scartati e ritorna
`_compiled(albero, spec_raccolta, registry)` — spec eventualmente vuota.

**Punto critico 1 — re-rooting di un `Tied`.** Quando si raccoglie la spec di un prefab sotto
un prefisso, il prefisso va applicato **sia all'optic target sia a ogni optic master** dentro
un `Tied`. Un `Tied` porta gli optic dei master: se si re-rootta solo il target, il tie
legge silenziosamente dal nodo sbagliato.

**Punto critico 2 — inversione del walker sugli operatori.** Il walker deve conoscere, per
ogni operatore, la mappa argomento-AST → campo dello struct. La composizione la **inverte**:
`a ∘ b` costruisce `Pipe(left=b, right=a)` (e `a |> b` → `Pipe(left=a, right=b)`). Se la
mappa è sbagliata, `cm.a.param` risolve silenziosamente al sottoalbero sbagliato — lo stesso
modo di fallire del re-rooting di un `Tied`. Gli operatori pointwise
(`Sum`/`Product`/`Difference`/`Quotient`) sono invece uniformi: 1° arg → `.left`, 2° → `.right`.

**Check d'identità (rete di sicurezza per il punto critico 2).** Dopo la costruzione, per
ogni foglia il builder verifica `get(albero, optic_foglia) === valore_scartato`. Costo
trascurabile e pagato solo a costruzione, quindi sempre attivo: se la mappa
argomento→campo del walker è sbagliata, l'errore scatta subito e con un messaggio chiaro,
invece di risolvere nomi sul sottoalbero sbagliato in silenzio.

### 3.6 — La macro `@constrain`

La macro produce solo la *ricetta*; tutto il resto avviene a runtime, perché la macro vede
solo il simbolo `model`, non il suo valore — quindi anche la validazione "esiste `narrow`?"
avviene lì, non all'espansione. La pipeline:

1. **Parse** di `@fix` / `@bound` / `@tie` / `@free` in voci `(path, kind, payload)`.
   Per `@tie` i master sono auto-rilevati: ogni riferimento `a.b.c` nell'espressione
   diventa un master (in ordine d'apparizione) e l'espressione una closure `f(m1, m2, …)`.
2. **Risoluzione** dei path contro la registry (errore con suggerimenti se un nome non
   esiste). Su un Model nudo singolo: registry implicita, parametri senza prefisso.
3. **Merge per target** sulla spec esistente, l'ultimo vince (override dei vincoli di
   fabbrica). `@free` produce una voce `Free` *esplicita*: è così che l'override funziona.
4. **Validazione sulla vista merged** — dopo il merge, non prima: con override e
   composizione un conflitto può emergere solo nella vista combinata (es.
   `@tie Hb.line.sigma = Ha.line.sigma` se un prefab aveva già legato quel target).
   - V1: nessun master di un `Tied` è a sua volta target `Tied` (niente catene);
   - V2: niente self-tie;
   - V3: bounds sani (`lo < hi`);
   - V4: valore corrente dentro i nuovi bounds (errore, non clamp silenzioso).
5. **Applicazione** dei valori di `@fix p = v` nell'albero, poi `_compiled(…)` → nuovo
   `CompiledModel` (I1 stabilita). Accetta sia un Model nudo sia un CompiledModel.

### 3.7 — Accesso e aggiornamento (`getproperty` / `@set`)

- `getproperty(cm, s)`: se `s` è nella registry ritorna un **`ComponentRef`** (cm radice +
  optic accumulato + sotto-registry). Su un ref, un sotto-componente approfondisce il ref;
  un nome di parametro risolve al valore via `optic ∘ PropertyLens(s)` sull'albero,
  riusando la macchina `Accessors`.
- `@set cm.narrow.amplitude = 3.0` si decompone nella ricorsione standard di `Accessors`
  in due passi, ed entrambi hanno ciò che serve:
  1. *interno* — `set` sul `ComponentRef`: il ref conosce il cm radice, quindi è qui che
     si controlla il vincolo del target (`Tied` → errore; `Bounded` → valida
     `lo ≤ v ≤ hi`, errore se fuori); ritorna il sotto-modello aggiornato;
  2. *esterno* — `set` sul `CompiledModel`: rimpiazza il sottoalbero all'optic della
     registry (stesso tipo concreto richiesto), poi `_compiled(…)` → **ri-risoluzione dei
     tie** (il parametro toccato può essere un master) → nuovo CompiledModel.
- **Collisione nomi.** I campi interni (`model`, `spec`, `names`) sono **riservati**: non
  usabili come nomi di componente (lo controlla `@model`). Resta sempre disponibile
  l'accesso esplicito `cm[:narrow]` (indicizzazione per simbolo, stile astropy) come via
  non ambigua.
- **`@set` su un `Tied` è errore**: un dipendente è calcolato dai master, scriverci un
  valore non ha senso. Aggiornabili `Free`/`Bounded`/`Fixed` (su un `Fixed` aggiorna il
  valore memorizzato: il parametro resta fisso).

### 3.8 — Casi limite (politica decisa)

| Caso | Comportamento |
|---|---|
| `@tie` e `@fix` sullo stesso target nello stesso blocco | merge: l'ultimo vince (coerente con l'override) |
| Master di un tie che è `Fixed` | permesso: il tie legge il valore memorizzato |
| Stesso oggetto prefab usato due volte con nomi diversi | ok: immutabile, i valori sono copiati nei due sottoalberi |
| `@set` su un `Tied` | errore |
| `@set` fuori bounds su un `Bounded` | errore (mai clamp silenzioso) |
| Compound costruito a mano (senza `@model`) passato a `@constrain` | errore con hint: "usa `@model` per dare i nomi" |
| Operatori su CompiledModel fuori da `@model` | errore informativo (mai perdita silenziosa di spec) |
| `2 * model`, `model + 1`, `-model` | errore informativo: "usa un `Const1D` nominato" |

### 3.9 — Assunzioni sul motore — verificate ✓

Il design era stato chiuso sul documento, senza rileggere `src/`. Verificate (e in
parte implementate) nella sessione sul rendering del fit loop:

- **A1 ✓** — le utility di §3.1 esistono con la semantica attesa (`src/params.jl`);
- **A2 ✓** — la spec del motore è una `NTuple` di `(optic, constraint)`;
- **A3 ✓** — il rebuild risolveva già i tie in seconda passata: `resolve(model, spec)`
  è stato *estratto* dal motore come funzione pubblica, non riscritto;
- **A4 ✓** — risolta con la doppia forma: `Fixed()` marker puro (fissa al valore
  corrente dell'albero, servirà a `@fix p`) e `Fixed(v)` che `compile` scrive
  nell'albero una volta — da lì in poi l'albero è l'unica fonte di verità;
- **A5 ✓** — le foglie sono struct immutabili coi parametri come campi, callable pointwise.

**Stato implementazione (12 giugno 2026):** il layer di authoring di §3.5–3.8 è fatto —
`@model` (forma a blocco e inline, builder a closure, assorbimento prefab con
re-rooting di spec e master dei `Tied`, check d'identità, nomi riservati/collisioni,
parametri `Free` di default), `@constrain` (merge per target con override "ultimo
vince", validazioni V1–V4 sulla vista merged, Model nudo con registry implicita),
registry `names` gerarchica (`Registry` per i prefab), `ComponentRef` con path di
lettura e `@set` (controlli `Tied`/`Bounded`, ri-risoluzione I1), `cm[:nome]`,
errori informativi (scalari nudi, algebra su CompiledModel fuori da `@model`,
compound costruiti a mano). Verificata la lista di §3.12 (119 test, hot path
zero-alloc e `@inferred` anche su modelli authored). Manca solo `fit` (§3.10,
design rimandato a documento dedicato).

### 3.10 — `fit` (segnaposto)

`fit(cm, data)` con firma minima, che ritorna un nuovo `CompiledModel` coi valori fittati.
I dettagli (optimizer, loss, pesi/incertezze, formato dei dati) sono rimandati a un design
dedicato.

### 3.11 — Documentazione di dominio

- `CONTEXT.md` con il glossario (Model, Component, Compound model, Constraint, Tied, Spec,
  CompiledModel, Prefab) e le relazioni.
- ADR per le cinque decisioni difficili da invertire:
  1. *I valori vivono nel modello* (no oggetto Params separato; no `render`).
  2. *Nomi di componente espliciti via `@model`* (non suffissi posizionali).
  3. *I vincoli dei prefab viaggiano namespaced*; `@model`/`@constrain` condividono il tipo
     `CompiledModel` perché ci sia dove assorbirli.
  4. *Risoluzione eager dei tie* (invariante I1: l'albero dentro un CompiledModel è sempre
     risolto; la valutazione non consulta mai la spec).
  5. *La chiamata è valutazione; la ricostruzione è `withparams`* (niente callable `pm(p)`:
     `cm(vettore)` sarebbe ambiguo).

### 3.12 — Verifica

- **Round-trip authoring**: `@model` + `@constrain` produce lo stesso `CompiledModel` che si
  otterrebbe a mano con `compile(model, spec)` (stessa `nfree`/`paramvector`, stesso
  risultato di `withparams`).
- **Override** (§3.6): da un prefab, restringere un bound di fabbrica e usare `@free`;
  verificare merge per-nome e che gli altri vincoli restino.
- **Composizione prefab**: `cont + Ha + Hb` porta `Ha.line.sigma`/`Hb.line.sigma` senza
  collisioni e `@constrain` li indirizza.
- **Indirizzamento attraverso gli operatori** (verifica l'inversione del walker): un
  componente nominato dentro un `Pipe` (`inner |> outer`) — assertare che `cm.inner`/`cm.outer`
  colpiscano il sottoalbero giusto. Più associatività annidata: `cont + line - absor` →
  `cont` a `.left.left`, `line` a `.left.right`, `absor` a `.right`.
- **Check d'identità di `@model`**: un test lo fa scattare ad arte (mappa del walker
  invertita) per garantire che il guardrail funzioni davvero.
- **`@tie` attraverso la composizione (caso più rischioso)**: un prefab il cui vincolo è un
  `@tie` (target *e* master), composto in un modello più grande; assertare che il tie si
  risolva sui master **prefissati**. Più: `@tie` verso un target già `Tied` nella spec merged
  → errore atteso.
- **Invariante I1**: dopo `@set` su un *master*, il valore del dipendente nel nuovo
  CompiledModel è già aggiornato, senza chiamare nulla (`cm2.broad.mean == cm2.narrow.mean`).
- **`@set`**: su un `Free`/`Bounded` → nuovo CompiledModel corretto; su un `Tied` → errore;
  fuori bounds su un `Bounded` → errore.
- **Hot path di `withparams`**: `@allocated withparams(cm, p) == 0` e `@inferred`, con e
  senza `Tied` nella spec.
- **Ambiguità eliminata**: chiamare un CompiledModel con un vettore non ricostruisce più dal
  vettore piatto (MethodError o errore informativo); `withparams` sì.
- **Regressione engine**: la suite `@testitem` passa dopo le rinominazioni (hot-path
  type-stable, `@allocated == 0`, autodiff attraverso `Tied`). Eseguire con TestItemRunner.
