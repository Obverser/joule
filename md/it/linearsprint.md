# ML Sprint: Marzo 17
### Regressione Lineare con GPA degli Studenti

> Aspetta! Questa pagina è stata tradotta principalmente con GPT--potrebbero esserci degli errori!

Ciao a tutti! Sta settimana, ci è stato dato un piccolo dataset e ci è stato chiesto di scrivere un modello in meno di 2 ore. Dovevamo trovare la correlazione, se presente, tra l'uso della biblioteca da parte degli studenti e il loro GPA. I nostri dati consistevano in 9 gruppi, che vanno da **meno di 2,00** a **più di 3,74**. Questo è completato dal rapporto degli studenti che utilizzano la biblioteca in quel gruppo.

| GPA       | Rapporto | Inutilizzato | Utilizzato |
|-----------|-------|--------|------|
| >3.74     | 67%   | 5368   | 11023|
| 3.50-3.74 | 66%   | 5403   | 9674 |
| 3.25-3.49 | 61%   | 5972   | 9474 |
| 3.00-3.24 | 56%   | 7019   | 9062 |
| 2.75-2.99 | 53%   | 5555   | 6260 |
| 2.50-2.74 | 49%   | 5577   | 5374 |
| 2.25-2.49 | 45%   | 4201   | 3489 |
| 2.00-2.24 | 42%   | 3653   | 2666 |
| <2.00     | 34%   | 18059  | 4126 |

Probabilmente hai già notato che il rapporto degli studenti che utilizzano la biblioteca diminuisce man mano che il GPA diminuisce. Ma come possiamo trovare la linea tra questi punti? Qui è dove possiamo applicare <u>Regressione Lineare</u>. Puoi pensare alla regressione lineare come un modo per trovare la *linea di miglior adattamento*-gradualmente adattando una linea per avvicinarsi a ogni punto in modo equo.

Quindi, innanzitutto, cosa stiamo cercando? Qual è *X* e *Y* del nostro grafico? In generale, vogliamo vedere come il rapporto degli utenti della biblioteca influisce sul bracket del GPA. Possiamo vedere come la percentuale tende verso il basso quando il GPA si avvicina a **2,00**. I nostri dati sono divisi in bin di **0,25 GPA** e gli estremi generalizzati per **oltre 3,74** e **sotto 2,00**. Sapendo questo, possiamo scegliere il GPA come asse *X* e il rapporto degli utenti della biblioteca come valore *Y*.

Prima di approfondire troppo, è importante notare che perderemo precisione quando uno studente si avvicina agli estremi dei nostri dati. Ho menzionato in precedenza che i dati hanno <u>estremi generalizzati</u>, poiché abbiamo **>** e **<**. Per il lato superiore, non dovrebbe essere un problema poiché il GPA di **4,00** è il limite superiore nella maggior parte dei casi. Tuttavia, **2,00** e meno sono tutti raggruppati in uno: non avremo un modello altrettanto preciso nei limiti inferiori.

Ora, come dovremmo implementarlo?

### Programmazione con librerie

Poiché il limite di tempo era di 2 ore, useremo librerie predefinite per questo. Puoi farlo con la pura matematica, ma richiede molto più tempo. Ho scelto di scrivere il modello utilizzando Rust. La libreria <u>dfdx</u> è stata utilizzata per la modellizzazione.

Prima di tutto, inseriamo i nostri dati in Rust. Poiché ci sono stati dati intervalli di GPA, dovremo fare compromessi su come inserire i nostri dati. Ho scelto di prendere semplicemente il GPA mediano in un intervallo dato e usarlo per la *componente X*. La *componente Y* è semplicemente la percentuale. Useremo il modulo Lineare.

```rust
use dfdx::{
    nn::builders::*,
    optim::{Optimizer, Sgd, SgdConfig},
    tensor::{Trace, TensorFrom},
    tensor_ops::{Backward, MeanTo},
};
use rand::prelude::*;

#[cfg(not(feature = "cuda"))]
type Dispositivo = dfdx::tensor::Cpu;

#[cfg(feature = "cuda")]
type Dispositivo = dfdx::tensor::Cuda;

type Modello = Linear<1, 1>;

let (x_dati, y_dati): ([[f32; 1]; 9], [[f32; 1]; 9]) = (
    [[1.875], [2.125], [2.375], [2.625], [2.875], [3.125], [3.375], [3.625], [3.875]], // X
    [[0.34], [0.42], [0.45], [0.49], [0.53], [0.56], [0.61], [0.66], [0.67]] // Y
);

fn main() {
    let dispositivo = Dispositivo::default();
    let mut modello = device.build_module::<Modello, f32>();

    /* ... */
```

Ora dobbiamo considerare come stiamo valutando il nostro modello. Per fare questo, stabiliamo le nostre perdite. Ho deciso di usare l'errore quadratico medio, ma ci sono diverse modalità per calcolare le perdite per adattarsi a qualsiasi ambito. L'errore quadratico medio, come suggerisce il nome, trova la distanza media tra il nostro punto previsto e il punto reale (l'errore). Matematicamente, sarebbe qualcosa del genere:

$$
\frac{1}{n} \sum (y_r - y_p)^2
$$

e dove $y_r$ rappresenta il nostro punto reale e $y_p$ la nostra previsione. dfdx fornisce una funzione per l'errore quadratico medio, ma per amor di chiarezza la  *creeremo noi stessi* .