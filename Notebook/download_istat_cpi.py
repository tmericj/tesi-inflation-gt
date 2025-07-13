# Script per scaricare dati ISTAT con analisi preliminare della DSD
import pandasdmx
from pandasdmx import Request
import pandas as pd
import logging
import sys

# Configura il logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Funzione principale
def fetch_istat_cpi_data():
    try:
        logger.info("Inizializzazione del client ISTAT...")
        istat = Request('ISTAT')

        # 1. Verifica disponibilità dataflow
        logger.info("FASE 1: Verifica disponibilità dataflow 'SDDS_PLUS_CPI_DF'...")
        try:
            available_dataflows = istat.dataflow() # Richiede tutti i dataflow
            # available_dataflows è un oggetto Message. I dataflow sono in .dataflow
            if 'SDDS_PLUS_CPI_DF' not in available_dataflows.dataflow:
                logger.error("Dataflow 'SDDS_PLUS_CPI_DF' non trovato nell'elenco dei dataflow disponibili.")
                logger.info(f"Dataflow disponibili che contengono 'CPI' o 'PRICE':")
                for key in available_dataflows.dataflow:
                    if 'CPI' in key.upper() or 'PRICE' in key.upper():
                        logger.info(f"- {key}")
                return False
            logger.info("Dataflow 'SDDS_PLUS_CPI_DF' trovato.")
        except Exception as e:
            logger.error(f"Errore durante il recupero dell'elenco dei dataflow: {e}")
            logger.error("Questo potrebbe indicare un problema generale con il servizio SDMX ISTAT o con la connettività.")
            return False

        # 2. Recupero e analisi della DSD
        dsd = None
        ordered_dimensions = []
        logger.info("\nFASE 2: Recupero della Data Structure Definition (DSD) per 'SDDS_PLUS_CPI_DF'...")
        try:
            dsd_response = istat.datastructure('SDDS_PLUS_CPI_DF')
            # dsd_response è un oggetto Message. La DSD è in .structure
            # Ci possono essere più strutture, prendiamo la prima (solitamente l'unica per un ID specifico)
            if dsd_response.structure:
                dsd = list(dsd_response.structure.values())[0] 
                logger.info(f"DSD ID: {dsd.id}, Nome: {dsd.name.get('it', dsd.name.get('en', 'N/A'))}")
                
                logger.info("Dimensioni definite nella DSD (nell'ordine restituito):")
                # Le dimensioni sono in dsd.dimensions (oggetto DimensionDescriptor)
                # dsd.dimensions.components è una lista di oggetti Dimension
                for dim in dsd.dimensions.components:
                    dim_id = dim.id
                    dim_name = dim.name.get('it', dim.name.get('en', 'N/A'))
                    ordered_dimensions.append(dim_id)
                    logger.info(f"- ID: {dim_id}, Nome: {dim_name}, Posizione: {dim.order}")
                    # Possiamo anche vedere le codelist associate, se necessario
                    # if dim.local_representation and dim.local_representation.enumerated:
                    #     logger.info(f"    Codelist ID: {dim.local_representation.enumerated.id}")

                logger.info(f"Ordine delle dimensioni rilevato dalla DSD: {ordered_dimensions}")
            else:
                logger.warning("Nessuna DSD trovata nella risposta per 'SDDS_PLUS_CPI_DF'.")
                logger.warning("Si procederà con un ordine di parametri standard, ma potrebbe fallire se l'ordine è cambiato.")

        except pandasdmx.api.ResourceNotFound:
            logger.error(f"DSD per 'SDDS_PLUS_CPI_DF' non trovata (ResourceNotFound).")
            logger.error("Questo è un problema: senza DSD, non possiamo essere sicuri della struttura della query.")
            logger.warning("Si procederà con un ordine di parametri standard, ma è probabile che fallisca.")
        except Exception as e:
            logger.error(f"Errore durante il recupero della DSD: {e}")
            logger.warning("Si procederà con un ordine di parametri standard, ma potrebbe fallire.")
            
        # 3. Tentativo di scaricare i dati
        logger.info("\nFASE 3: Tentativo di scaricare i dati CPI...")
        
        # Parametri base della query
        base_key_params = {
            'FREQ': 'M',            # Frequenza mensile
            'REF_AREA': 'IT',       # Italia
            'MEASURE': 'I',         # Indice (non variazione %)
            'PRICE': 'CP00',        # Totale generale (NIC)
            'UNIT_MEASURE': 'INX',  # Indice (base 2015=100)
        }
        
        # Costruisci la chiave 'key' per la richiesta dati.
        # Se abbiamo ottenuto le dimensioni dalla DSD, proviamo a rispettare quell'ordine.
        # Nota: pandasdmx di solito gestisce bene un dizionario per 'key',
        # ma per forzare un ordine specifico per l'URL REST, una stringa potrebbe essere più diretta
        # se sapessimo l'esatto formato atteso dal server ISTAT (es. 'M.IT.I.CP00.INX.SA').
        # Per ora, ci affidiamo a pandasdmx per costruire la key dal dizionario.
        # L'avviso ISTAT menziona "posizione delle dimensioni", che è cruciale per le query REST.
        
        # Tentiamo prima con dati destagionalizzati (SA), poi grezzi
        adjustments_to_try = {
            'SA': 'SA', # Destagionalizzato
            'RAW': None # Non specificato (dovrebbe dare dati grezzi)
        }
        
        data_found = False
        final_data_df = None
        successful_adjustment_type = None

        for adj_type, adj_value in adjustments_to_try.items():
            current_key_params = base_key_params.copy()
            if adj_value:
                current_key_params['ADJUSTMENT'] = adj_value
            
            # Se non abbiamo un ordine dalla DSD, pandasdmx userà l'ordine del dizionario (o il suo interno)
            # Se abbiamo un ordine dalla DSD, potremmo voler costruire la chiave come stringa
            # seguendo quell'ordine. Es: key_string = ".".join([current_key_params[d] for d in ordered_dimensions if d in current_key_params])
            # Tuttavia, la gestione di `pandasdmx` con un dizionario di solito è sufficiente se le dimensioni sono corrette.
            # L'importante è che TUTTE le dimensioni obbligatorie della DSD siano presenti nella chiave.

            logger.info(f"Tentativo con parametri ({adj_type}): {current_key_params}")
            
            try:
                resp = istat.data(
                    resource_id='SDDS_PLUS_CPI_DF',
                    key=current_key_params, # Passiamo il dizionario
                    params={'startPeriod': '2004-01', 'endPeriod': '2024-12'}
                )

                if not resp.data: # resp.data è una lista di DataSet
                    logger.warning(f"Nessun dataset trovato nella risposta per {adj_type}.")
                    continue
                
                data_df = resp.to_pandas()

                if data_df.empty:
                    logger.warning(f"Dataframe vuoto restituito per {adj_type}.")
                    continue
                
                logger.info(f"Dati ({adj_type}) ottenuti con successo!")
                final_data_df = data_df
                successful_adjustment_type = adj_type
                data_found = True
                break # Dati trovati, usciamo dal loop

            except pandasdmx.api.ResourceNotFound as rnfe:
                logger.error(f"ResourceNotFound durante la richiesta dati per {adj_type}: {rnfe}")
                logger.error("Questo potrebbe significare che la combinazione di chiavi specificata non esiste.")
            except Exception as e:
                logger.error(f"Errore durante la richiesta dati per {adj_type}: {e}")
                # import traceback # Per debug più approfondito
                # logger.debug(traceback.format_exc())
        
        if not data_found:
            logger.error("Impossibile scaricare i dati CPI dopo tutti i tentativi.")
            logger.error("Possibili cause:")
            logger.error("1. Problemi temporanei con il servizio ISTAT.")
            logger.error("2. Modifiche alla struttura delle query (ordine/nome dimensioni) non gestite correttamente.")
            logger.error("   Se la DSD è stata recuperata, verifica che tutti i parametri chiave corrispondano.")
            logger.error("   Se la DSD non è stata recuperata, è difficile diagnosticare ulteriormente.")
            logger.error("3. Il 'tool excel' menzionato da ISTAT potrebbe essere necessario per adattare le query.")
            return False

        # 4. Elaborazione e salvataggio
        logger.info(f"\nFASE 4: Elaborazione dati ({successful_adjustment_type})...")
        
        # Il DataFrame restituito da to_pandas() può essere una Serie o un DataFrame,
        # con un MultiIndex. Resettiamo l'indice per avere le dimensioni come colonne.
        if isinstance(final_data_df, pd.Series):
            processed_df = final_data_df.reset_index()
            # La serie ha un nome che diventa la colonna valore, o si chiama 'value' o 0
            # Cerchiamo di rinominarla in 'Value'
            if 0 in processed_df.columns and len(processed_df.columns) > 1 and isinstance(processed_df.columns[-1], int):
                 processed_df.rename(columns={processed_df.columns[-1]: 'Value'}, inplace=True)
            elif final_data_df.name:
                 processed_df.rename(columns={final_data_df.name: 'Value'}, inplace=True)
            # Se non ha nome e non è 0, potrebbe già essere 'value' o 'OBS_VALUE'
        else: # è un DataFrame
            processed_df = final_data_df.reset_index()

        logger.info(f"Colonne dopo reset_index: {processed_df.columns.tolist()}")

        # Identifica colonna tempo e valore
        time_col_name = None
        value_col_name = None

        # Nomi comuni per la colonna tempo in SDMX
        time_candidates = ['TIME_PERIOD', 'Time', 'time_period']
        for candidate in time_candidates:
            if candidate in processed_df.columns:
                time_col_name = candidate
                break
        if not time_col_name: # Fallback se non trova nomi standard
            # Cerca una colonna che assomigli a un periodo o data
            for col in processed_df.columns:
                if 'TIME' in col.upper() or 'PERIOD' in col.upper() or 'DATE' in col.upper():
                    time_col_name = col
                    break
            if not time_col_name:
                 logger.warning("Colonna tempo non identificata univocamente, si usa la prima colonna che non è 'Value'.")
                 # Questo è un fallback rischioso, la DSD dovrebbe chiarire
                 for col in processed_df.columns:
                     if col not in ['Value', 'OBS_VALUE']: # Evita la colonna valore
                         time_col_name = col
                         break


        # Nomi comuni per la colonna valore
        value_candidates = ['OBS_VALUE', 'Value', 'value'] # 'Value' è spesso il default di to_pandas
        for candidate in value_candidates:
            if candidate in processed_df.columns:
                value_col_name = candidate
                break
        if not value_col_name: # Fallback
            # Cerca una colonna numerica che non sia una dimensione nota
            known_dims_ids = [d.upper() for d in base_key_params.keys()]
            potential_value_cols = [
                col for col in processed_df.columns 
                if processed_df[col].dtype in ('float64', 'int64') and col.upper() not in known_dims_ids and col != time_col_name
            ]
            if potential_value_cols:
                value_col_name = potential_value_cols[0]
            else: # Ultima risorsa
                value_col_name = processed_df.columns[-1]


        if not time_col_name or not value_col_name:
            logger.error("Impossibile identificare colonna tempo o valore nel DataFrame risultante.")
            logger.info("Colonne disponibili: " + str(processed_df.columns.tolist()))
            logger.info("DataFrame head:\n" + str(processed_df.head()))
            return False

        logger.info(f"Colonna tempo identificata: '{time_col_name}', Colonna valore: '{value_col_name}'")
        
        # Seleziona e rinomina
        df_to_save = processed_df[[time_col_name, value_col_name]].copy()
        df_to_save.columns = ['Time', 'Value']

        # Assicura che la colonna 'Time' sia in formato Periodo per l'ordinamento
        try:
            df_to_save['Time'] = pd.to_datetime(df_to_save['Time']).dt.to_period('M')
        except Exception as e:
            logger.warning(f"Conversione colonna 'Time' a Periodo fallita: {e}. L'ordinamento potrebbe non essere cronologico.")

        df_to_save.sort_values('Time', inplace=True)
        
        output_filename = f"CPI_NIC_{successful_adjustment_type}_Italy_2004_2024_DSD_Attempt.csv"
        df_to_save.to_csv(output_filename, index=False)
        logger.info(f"Dati salvati con successo in: {output_filename}")
        print(f"\nPrime righe del DataFrame salvato ({output_filename}):")
        print(df_to_save.head())
        return True

    except Exception as e:
        logger.error(f"Errore generale nello script: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == '__main__':
    if fetch_istat_cpi_data():
        logger.info("Script completato con successo.")
    else:
        logger.error("Script terminato con errori.")

