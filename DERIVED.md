# Piano: Derived Quantities

Stato: proposta, non implementata.
Scope di questo documento: **solo la creazione** delle quantità derivate (tipo, macro,
storage, valutazione, validazione). Integrazione con catene MCMC e ottimizzatore è
esplicitamente **fuori scope** — vedi "Fuori scope" in fondo.

## Cosa

Una **Derived Quantity** è una quantità nominata calcolata dai valori del modello
ricostruito. Non occupa slot nel vettore piatto dei parametri, non entra mai nel render,
e non ha un campo di struct dove atterrare (altrimenti sarebbe un `Tied`).

```julia
m = @model begin
    g = Gaussian1D(amplitude = 1.0, mean = 0.0, sigma = 1.0)
    g
end

@derive m.flux := m.g.amplitude * m.g.sigma * sqrt(2π)
@derive m.fwhm := 2.355 * m.g.sigma

derived(m)                        # (flux = 2.5066…, fwhm = 2.355)
derived(withparams(m, [2.0, 0.5, 1.5]))   # ricalcolate sui nuovi parametri
derivednames(m)                   # [:flux, :fwhm]
nderived(m)                       # 2
```

## Decisioni di design

### D1 — le derivate si leggono dall'albero ricostruito, non dagli slot di `p`

`Tied` legge i locali `_pN` (slot del vettore piatto) perché partecipa alla
*costruzione* dell'albero dentro `withparams`; per questo `validate` impone che i suoi
master siano Free/Bounded ([constrain.jl:47](src/constrain.jl:47)).

`Derived` consuma l'albero a costruzione **finita**: `cm.g.model.amplitude` contiene già
il valore giusto qualunque sia il constraint. Conseguenze:

- i master possono essere **Free, Bounded, Fixed o Tied** — nessuna restrizione;
- la regola "no catene" non si applica (nessun ordine di valutazione da garantire);
- una Derived può referenziare un'altra Derived? **No** in questa iterazione: i path sono
  `(leaf, field)` sull'albero, e una Derived non è un leaf. Vedi "Non fatto".

### D2 — definizioni portate attraverso `withparams`, valori calcolati su richiesta

`withparams` copia il campo `derived` nel `CompiledModel` ricostruito. Non materializza
i valori.

Dal punto di vista dell'utente le derivate sono "aggiornate assieme alle altre":
`derived(withparams(m, p))` è sempre coerente con `p`. Quello che si evita è pagare il
calcolo a **ogni** chiamata dell'obiettivo — `chi2` gira migliaia di volte nel loop
interno dell'ottimizzatore, e nessuno legge le derivate fino a fine fit.

Alternativa scartata: materializzare i valori dentro il `CompiledModel` restituito. Stessa
UX, costo nel hot path. Se servisse, è una modifica alla stessa riga della `@generated`.

**Costo accettato**: `D` nel parametro di tipo significa una specializzazione di
`withparams` per ogni insieme distinto di derivate. Il corpo emesso ignora `D` (legge solo
`cm.parameters[1]`), ma Julia specializza comunque: due `@derive` in sessione interattiva
= 3 tipi `CompiledModel` = 3 corpi `withparams` identici compilati. È il prezzo di un
`derived()` type-stable, e si paga in latenza di compilazione, non a runtime.

### D3 — `:=` è l'operatore dedicato

Le quantità derivate si dichiarano con `:=`, distinto dagli operatori delle altre
operazioni sul modello:

| operatore | significato |
|---|---|
| `=`  | `@fix` — parametro pinnato a una costante |
| `->` | `@tie` — parametro calcolato da altri parametri, atterra in un campo |
| `in` | `@bound` — parametro vincolato a un intervallo |
| `~`  | `@prior` — distribuzione a priori |
| `:=` | `@derive` — quantità derivata, **non** atterra in nessun campo |

Verificato su Julia 1.12.6: `:=` è parsabile e produce `Expr(Symbol(":="), lhs, rhs)`;
`MacroTools.@capture(e, root_.name_ := rhs_)` lo cattura; il macro lo riceve prima del
lowering. Julia non gli dà semantica propria, quindi **deve** stare dentro un macro —
fuori dà errore di sintassi a valutazione. Non è un problema (è sempre dentro
`@derive`/`@constrain`), ma va detto nella docstring perché il messaggio d'errore nativo
non nomina AstroFit.

Da verificare in implementazione: che [Runic.jl](https://github.com/fredrikekre/Runic.jl)
non si strozzi su `:=` nei sorgenti e nei test.

### D4 — le derivate non entrano nella camminata degli slot

`_slotmap!`, `params`, `bounds`, `paramnames`, `nfree` camminano l'albero nello stesso
ordine DFS e devono restare allineati. `derived` vive **fuori** dall'albero, in un campo
separato di `CompiledModel`. Nessuna di quelle funzioni cambia. Invariante da testare
esplicitamente (D4).

### D5 — mai durante il render

Le derivate non toccano `render`/`_eval`/`chi2`. Motivo (già stabilito, da registrare in
ADR-0007):

- `_eval` sul ramo pointwise restituisce un `Broadcasted` non materializzato
  ([render.jl:69](src/render.jl:69)); scrivere un output laterale lì significa rompere la
  fusione su cui poggia il benchmark ≤1.0x;
- i duali ForwardDiff attraversano `withparams` → `render` (ADR-0005), e un buffer
  `Float64` pre-allocato scarterebbe silenziosamente la derivata;
- `chi2` non materializza mai un array; un hook nel render lo costringerebbe.

## Modifiche, file per file

### `src/constraints.jl` — nuovo tipo

`Derived` **non** è un `AbstractConstraint` (non vincola nulla, non sta in un
`constraints` tuple). Stessa *forma* di `Tied`, semantica diversa.

```julia
# value = f(path₁ … pathₙ), letto dall'albero ricostruito.
# Paths nel tipo così la valutazione è statica, come per Tied (decisione 8).
struct Derived{Paths, F}
    f::F
end
Derived(paths::Tuple, f::F) where {F} = Derived{paths, F}(f)
```

Alternativa: file nuovo `src/derived.jl`. Preferibile se `derived`/`derivednames` e la
validazione ci finiscono dentro — tiene il concetto in un posto solo. Da decidere in
implementazione; il resto del piano non cambia.

### `src/compiled.jl` — terzo campo

```julia
struct CompiledModel{T, P, D}
    tree::T
    priors::P
    derived::D          # NamedTuple{names, <:Tuple{Vararg{Derived}}}
end
CompiledModel(tree, priors) = CompiledModel(tree, priors, NamedTuple())
```

Il costruttore a due argomenti mantiene funzionanti i **5 siti di costruzione** esistenti
senza toccarli.

`getproperty` va aggiornato: oggi lascia passare `:tree` e `:priors` e manda tutto il
resto a `_nav` ([compiled.jl:27](src/compiled.jl:27)). Aggiungere `:derived` alla lista
dei campi.

> **Nota**: `m.flux` **non** restituisce il valore derivato. `getproperty` resta
> navigazione dei leaf; le derivate si leggono con `derived(m).flux`. Mescolare i due
> spazi di nomi renderebbe ambiguo `m.qualcosa` e costringerebbe a un controllo di
> collisione leaf↔derived a ogni `@derive`. Vedi "Non fatto".

### `src/render.jl` — una firma

```julia
evalstyle(::Type{CompiledModel{T, P}}) where {T, P} = evalstyle(T)   # riga 48
# →
evalstyle(::Type{<:CompiledModel{T}}) where {T} = evalstyle(T)
```

`src/priors.jl:35,41` usano `CompiledModel{<:Any, Nothing}`: la parametrizzazione parziale
in Julia matcha qualunque `D`, **restano invariati**.

### `src/withparams.jl` — una riga

```julia
return Expr(:block, loads..., :(CompiledModel($tree, getfield(cm, :priors))))
# →
return Expr(:block, loads..., :(CompiledModel($tree, getfield(cm, :priors), getfield(cm, :derived))))
```

Copia di campo, nessun calcolo. Da confermare col benchmark, non da asserire (vedi
"Verifica").

### `src/constrain.jl` — propagazione

`setconstraint` e `resetconstraints` ricostruiscono `CompiledModel(tree, priors)`
([constrain.jl:19,31](src/constrain.jl:19)). Devono portarsi dietro `derived`, altrimenti
un `@fix` cancella silenziosamente le quantità derivate. **Questo è il bug più probabile
dell'intera feature** — un test dedicato, non solo una riga di codice.

Nota semantica: `resetconstraints` azzera i constraint ma **non** le derivate. `@constrain`
dichiara lo stato dei vincoli, non delle quantità derivate; sono spazi ortogonali.

### `src/derived.jl` (nuovo) — API di lettura

```julia
# @generated: i path stanno nel tipo, quindi la lettura si risolve a tempo di
# specializzazione in una catena di getfield dritta — stesso idioma di
# _treeexpr/_fieldexpr per Tied (withparams.jl:40).
@generated function _dvalue(cm, d::Derived{Paths}) where {Paths}
    reads = (:(getfield(getproperty(cm, $(QuoteNode(l))).model, $(QuoteNode(f))))
             for (l, f) in Paths)
    return :(d.f($(reads...)))
end

derived(cm::CompiledModel) = map(d -> _dvalue(cm, d), getfield(cm, :derived))
derivednames(cm::CompiledModel) = collect(keys(getfield(cm, :derived)))
nderived(cm::CompiledModel) = length(getfield(cm, :derived))
```

**Perché `@generated` e non un `map` con `Val`.** La versione ovvia
(`map(p -> _readpath(cm, Val(p)), Paths)`) *non* inferisce: `p` è un valore estratto da
una tupla, tutti gli elementi hanno tipo `Tuple{Symbol,Symbol}`, quindi il compilatore
vede `Val(::Tuple{Symbol,Symbol})` e produce `Val{<:Any}` → dispatch dinamico su
`_readpath`. Lo srotolamento non salverebbe nulla; la constant-propagation *potrebbe*
salvarlo, che è peggio di un fallimento netto — funziona nel test e degrada nel modello
di un utente. `@generated` è la stessa risposta che il package dà già allo stesso
problema per `Tied`.

`derived` restituisce una `NamedTuple`, che è la forma giusta sia per la stampa che per la
mappatura futura su una catena.

### `src/constrain.jl` — validazione

`validate` chiama oggi `_vnode` (regole dei tie) e `_validate_priors`
([constrain.jl:39](src/constrain.jl:39)). Aggiungere `_validate_derived(cm)`:

- ogni path `(leaf, field)` deve **esistere** (leaf presente nell'albero, field presente
  nella struct);
- nessun requisito di freeness — è la differenza con `_vnode`, ed è deliberata (D1).

Messaggio d'errore sul modello di quello dei tie: nomina la derivata, il path e il motivo.

### `src/macro.jl` — `@derive`

`_tiewalk` ([macro.jl:103](src/macro.jl:103)) è riusabile **così com'è**: cammina la RHS,
sostituisce ogni `root.leaf.field` con un argomento fresco, raccoglie i path in ordine e
restituisce `(paths_expr, lambda)`. È esattamente ciò che serve.

```julia
macro derive(a)
    @capture(a, root_.name_ := rhs_) ||
        error("@derive expects `model.name := expression`")
    root isa Symbol || error("@derive expects a model variable on the left")
    (pe, lam) = _tiewalk(rhs, root)
    return :($(esc(root)) = validate(_setderived($(esc(root)), $(QuoteNode(name)),
                                                 Derived($pe, $(esc(lam))))))
end
```

`:=` rende il parsing più semplice di `@tie`: `@capture` estrae root, nome ed espressione
in un colpo, e non serve l'unwrapping del `:block` che `@tie` fa per `->`
([macro.jl:179](src/macro.jl:179)) — `->` costruisce una funzione anonima e ci mette un
blocco dentro, `:=` no.

Con:

```julia
_setderived(cm::CompiledModel, name::Symbol, d::Derived) =
    CompiledModel(getfield(cm, :tree), getfield(cm, :priors),
                  merge(getfield(cm, :derived), NamedTuple{(name,)}((d,))))
```

Nessuna ambiguità con gli altri macro: l'operatore è diverso, e il path a sinistra ha
**due** livelli (`m.flux`) contro i tre di `@tie` (`m.g2.mean`). Auto-rebinding della
variabile modello identico agli altri macro.

**Forma a blocco dentro `@constrain`.** `:=` è auto-identificante, quindi entra nel
dispatch del blocco di `@constrain` ([macro.jl:317](src/macro.jl:317)) senza ambiguità
accanto alle forme esistenti:

```julia
@constrain m begin
    g.amplitude in (0.0, 10.0)
    g2.mean -> g1.mean + 0.5
    flux := g.amplitude * g.sigma * sqrt(2π)
end
```

Attenzione a una differenza semantica: `@constrain` chiama `resetconstraints` all'inizio
per dichiarare lo stato **completo** dei vincoli ([constrain.jl:30](src/constrain.jl:30)).
Le derivate sono uno spazio ortogonale e **non** vengono azzerate — un `@constrain` senza
`:=` non cancella le derivate esistenti. Da decidere se questa asimmetria è accettabile o
se il blocco deve dichiarare anche le derivate per intero; propendo per l'asimmetria
(azzerare le derivate a ogni riaggiustamento dei vincoli sarebbe sorprendente).

Rimane fuori dalla prima iterazione: il macro standalone si compone e basta a sé.

### `src/show.jl`

Mostrare le derivate col loro valore corrente in fondo al display del modello. Cosmetico,
ultimo passo, dopo che i test passano.

### `src/AstroFit.jl`

Export: `Derived`, `@derive`, `derived`, `derivednames`, `nderived`. Include del nuovo file.

## Ordine di implementazione

1. `Derived` + campo su `CompiledModel` + costruttore a due argomenti + `evalstyle`
   + **propagazione in `setconstraint`/`resetconstraints`**. Tutti i siti di costruzione
   di `CompiledModel` si sistemano in un colpo: separarli lascerebbe un intervallo in cui
   il terzo campo viene silenziosamente perso e nessun test può accorgersene (niente
   `@derive` fino al passo 5). "Suite verde" lì significherebbe *non testabile*, non
   *corretto*.
2. `withparams` porta il campo.
3. `derived`/`derivednames`/`nderived` + `_setderived`.
4. `_validate_derived`.
5. Macro `@derive`.
6. `show`.

I passi 1–2 sono puramente strutturali e vanno committati separati dal resto: se il
benchmark regredisce, si sa esattamente dove.

## Verifica

Nuovo `test/derived_tests.jl` (`@testitem`, stile `test/*_tests.jl`):

- `derived(m)` dà i valori attesi sul modello iniziale;
- `derived(withparams(m, p))` traccia `p` — il caso centrale;
- derivata che referenzia un parametro `Fixed` e uno `Tied`: funziona (D1);
- **`nfree`/`params`/`paramnames`/`bounds` invariati** dopo `@derive` (D4);
- **`derived` sopravvive a `@fix`/`@bound`/`@tie`/`@constrain`** (il bug del passo 1);
- path inesistente → `ArgumentError` che nomina la derivata e il path;
- `@inferred derived(m)` — guardia di regressione sulla `@generated` `_dvalue`;
- ForwardDiff: `derived(withparams(m, duals))` propaga la derivata (viene gratis da D1, ma
  è il tipo di cosa che si rompe in silenzio).

Benchmark: `benchpkg AstroFit --rev=main,dirty --bench-on=main` dopo il passo 2.
**Attenzione a cosa misura**: `benchmark/benchmarks.jl` non contiene modelli con derivate,
quindi quella corsa verifica solo "nessuna regressione quando la feature non è usata" — il
che è esattamente il claim ≤1.0x da proteggere, ma non dice nulla su un modello che ha
derivate. Se serve coprire anche quel caso, aggiungere una voce al `SUITE` con un modello
`@derive`d; altrimenti va scritto che la guardia copre solo il percorso senza derivate.

## Da registrare

- **ADR-0007** — "Derived quantities calcolate dall'albero ricostruito". Decisione: mai
  durante il render (D5); nessuno slot nel vettore piatto (D4); nessun vincolo sul
  constraint dei master (D1); definizioni portate, non materializzate (D2); `:=` come
  operatore dedicato (D3).
- **CONTEXT.md**, glossario:
  > **Derived Quantity**: una quantità nominata calcolata dai valori del modello
  > ricostruito, dichiarata con `:=`, che non occupa slot nel vettore piatto e non entra
  > nel render.
  > _Avoid_: generated quantity, extra parameter, tracked quantity, output,
  > pseudo-parameter.

  Relazioni: una Derived Quantity si distingue da un parametro Tied per l'assenza di un
  campo dove atterrare e per la libertà nei constraint dei master.

## Non fatto (deliberatamente)

- `m.flux` come accesso diretto al valore. Ambiguo con la navigazione dei leaf. Se servirà,
  serve un controllo di collisione dei nomi in `_setderived`.
- Derivate che referenziano altre derivate. Reintroduce l'ordine di valutazione, cioè
  esattamente la complessità che D1 elimina.
- Rimozione di una derivata (`@underive`). YAGNI: si ricostruisce il modello.
- `:=` dentro il blocco `@constrain`. Non ambiguo e a basso costo grazie all'operatore
  dedicato (vedi sezione macro), ma il macro standalone basta alla prima iterazione.

## Fuori scope di questo piano

Integrazione con i consumatori — da progettare **dopo** che il meccanismo esiste:

- catene MCMC (l'hook è `Pigeons.extract_sample`, forma a due argomenti; rischio noto:
  disallineamento d'ordine fra `extract_sample` e `sample_names`, da mitigare facendo
  derivare entrambi da una sola funzione);
- risultato dell'ottimizzatore (`derived(withparams(cm, sol.u))` funziona già senza
  aggiungere nulla, una volta esistente il meccanismo);
- `ObjectiveFunction` non viene toccata: nessun costo nel loop interno.
