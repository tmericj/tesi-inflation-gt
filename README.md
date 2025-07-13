# tesi-inflation-gt
Master's thesis on inflation analysis - Complete code and visualizations

# Predire l'Inflazione Italiana con Google Trends: Una Nuova Metodologia

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## Descrizione

Questo repository contiene il codice completo e le visualizzazioni della tesi magistrale **"Predire l'Inflazione Italiana con Google Trends: Una Nuova Metodologia"** (Anno Accademico 2024/2025).

La ricerca valuta se l'integrazione di indicatori derivati da Google Trends possa migliorare la capacità predittiva dei modelli econometrici tradizionali per l'inflazione italiana nel periodo 2004-2024.

### Obiettivi Principali

- Costruzione di indicatori complementari da dati Google Trends tramite PCA.
- Valutazione del contributo marginale di questi indicatori in modelli SARIMAX.
- Analisi della causalità bidirezionale tra comportamenti di ricerca online e inflazione.
- Test statistici per orizzonti temporali rilevanti per la politica monetaria (6-12 mesi).

### Risultati Chiave

- **Causalità di Granger**: L'indice inflazione GT anticipa il NIC di 4-16 mesi.
- **Miglioramenti statisticamente significativi**: Test Clark-West per orizzonti 6 e 12 mesi (p-value < 0.05).
- **Rotture strutturali**: Identificati outliers gennaio e ottobre 2022 (non marzo 2022 come inizialmente ipotizzato).


## Metodologia

### Costruzione degli Indicatori
- **9 gruppi tematici** di query basati su classificazione ECOICOP + categorie trasversali;
- **Principal Component Analysis (PCA)** a due livelli;
- **Due indici complementari**:
  - `indice_Inflazione_GT_PCA`: attenzione diretta all'inflazione;
  - `indice_Tematico_GT`: interesse per tematiche economiche correlate.

### Modelli Econometrici
- **Modello base**: SARIMAX(1,0,1)(0,0,0,12) con pulse dummy per outlier.
- **Modelli aumentati**: Integrazione indicatori GT con lag ottimali.
- **Errori standard robusti** (HC0) per la sola eteroschedasticità (serie stazionarie perché differenziate all'inizio).


## Struttura del Repository
```
tesi-inflation-gt/
│
├── 📂 Notebook/                                # Jupyter notebooks principali
│   ├──  PCA_GT_Query_ibrida_v6.ipynb           # Costruzione indicatori GT tramite PCA
│   ├──  Indice_NIC_generale.ipynb              # Unificazione serie storica NIC ISTAT
│   ├──  Destagionalizzazione_GT.ipynb          # Destagionalizzazione X-13-ARIMA-SEATS
│   ├──  Fase_2_staz_e_diff1.ipynb              # Test stazionarietà e differenziazione
│   ├──  Fase_3_CCF.ipynb                       # Analisi correlazioni incrociate (CCF)
│   ├──  Fase_4_Granger.ipynb                   # Test causalità di Granger
│   ├──  Fase_5_test_Chow.ipynb                 # Test rotture strutturali (Chow)
│   ├──  Fase_5.1_test_Chow_posteriori.ipynb    # Verifica a posteriori degli outlier identificati
│   ├──  SARIMAX_base.ipynb                     # Modello benchmark
│   ├──  SARIMAX_base_HAC.ipynb                 # Modello con errori robusti
│   ├──  SARIMAX_base_HAC_pulse.ipynb           # Test pulse dummy marzo 2022
│   ├──  SARIMAX_base_HAC_pulse_outlier.ipynb   # Modello finale con outlier
│   ├──  SARIMAX_e_GT.ipynb                     # Modelli aumentati con GT
│   ├──  SARIMAX_e_GT_v2.ipynb                  # Versioni alternative
│   ├──  SARIMAX_e_GT_v3.ipynb                  # Versioni alternative
│   └──  SARIMAX_e_GT_outofsample.ipynb         # Valutazione out-of-sample
│
├── 📂 ISTAT_data/                            # Dati ufficiali inflazione
├── 📂 PCA/                                   # Output analisi componenti principali
├── 📂 Analisi_Correlazione_Incrociata_(CCF)/ # Analisi CCF
├── 📂 Analisi_Rottura_Chow/                  # Test rotture strutturali
├── 📂 Destagionalized_Indexes/               # Serie destagionalizzate
├── 📂 First_Difference_indexes/              # Serie differenziate
├── 📂 SARIMAX_modelli/                       # Output modelli finali
├── 📂 Official_4/                            # Dati query Google Trends
├── 📂 HNKPC/                                 # Test Hybrid NKPC
├──–   MIDAS.ipynb                            # Test modelli MIDAS
├──–   LICENSE
│
└── 📄 README.md                              # Questo file
```



## Esecuzione Sequenziale
Costruzione indicatori GT:
- jupyter notebook Notebook/PCA_GT_Query_ibrida_v6.ipynb

Preparazione dati NIC:
- jupyter notebook Notebook/Indice_NIC_generale.ipynb

Analisi serie temporali (sequenza):
- jupyter notebook Notebook/Destagionalizzazione_GT.ipynb
- jupyter notebook Notebook/Fase_2_staz_e_diff1.ipynb
- jupyter notebook Notebook/Fase_3_CCF.ipynb
- jupyter notebook Notebook/Fase_4_Granger.ipynb

Modellazione econometrica:
- jupyter notebook Notebook/SARIMAX_base_HAC_pulse_outlier.ipynb
- jupyter notebook Notebook/SARIMAX_e_GT_outofsample.ipynb


## Contributi Metodologici

- **Approccio ibrido** per selezione query (ECOICOP + categorie trasversali).
- **PCA a due livelli** per separare segnale inflazione da dinamiche tematiche.
- **Benchmark robusto** con gestione outlier e errori HC0.
- **Valutazione completa** con test Clark-West per modelli nested.

## Citazioni

Se utilizzi questo codice per ricerca accademica, cita:

```bibtex
@mastersthesis{merici2025,
  title={Predire l'Inflazione Italiana con Google Trends: Una Nuova Metodologia},
  author={Tommaso Merici},
  school={Università Ca' Foscari Venezia},
  year={2025},
  type={Tesi di Laurea Magistrale},
  url={https://github.com/tmericj/tesi-inflation-gt}
}
```

## Autore
- Nome: Tommaso Merici
- Matricola: 873042
- Corso: Laurea Magistrale in Economia e Finanza
- Università: Ca' Foscari Venezia

## Relatore
- Prof. Davide Raggi

**Contatti**
* Email: tommaso.merici99@gmail.com
* LinkedIn: Tommaso Merici (https://www.linkedin.com/in/tommaso-merici-303b79293/)
* GitHub: @tmericj

**Licenza**: questo progetto è distribuito sotto licenza MIT.


## ❓ FAQ

**Q: I dati Google Trends sono inclusi nella repo?**  
A: Sì, nella cartella `Official_4/` trovi i dati estratti utilizzati nell'analisi.

**Q: Posso replicare l'analisi con dati più recenti?**  
A: Sì, dovrai aggiornare i download da Google Trends e ISTAT seguendo la metodologia descritta.

**Q: Quali sono i requisiti hardware minimi?**  
A: RAM 8GB+ consigliata per l'elaborazione delle serie temporali e PCA.