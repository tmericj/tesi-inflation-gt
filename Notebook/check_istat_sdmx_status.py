# Questo script verifica lo stato dell'endpoint principale dataflow di ISTAT SDMX.

import requests
import logging

# Configura il logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_istat_sdmx_status():
    """
    Controlla lo stato dell'endpoint principale dataflow di ISTAT SDMX.
    """
    url = "https://sdmx.istat.it/SDMXWS/rest/dataflow/all/latest"
    logger.info(f"Tentativo di connessione a: {url}")
    try:
        response = requests.get(url, timeout=30) # Timeout dopo 30 secondi
            
        if response.status_code == 200:
            logger.info(f"Successo! Codice di stato: {response.status_code}. Il servizio ISTAT SDMX sembra essere OPERATIVO.")
            logger.info("Puoi provare a eseguire nuovamente lo script di download completo.")
            # logger.info(f"Contenuto ricevuto (prime 500 battute): {response.text[:500]}")
            return True
        else:
            logger.error(f"Errore dal server ISTAT. Codice di stato: {response.status_code}")
            logger.error(f"Risposta del server (prime 500 battute): {response.text[:500]}")
            logger.error("Il servizio ISTAT SDMX NON sembra essere operativo al momento.")
            return False
    except requests.exceptions.RequestException as e:
        logger.error(f"Errore di connessione durante il tentativo di raggiungere il server ISTAT: {e}")
        logger.error("Verifica la tua connessione internet o riprova più tardi. Potrebbe anche essere un problema del server ISTAT.")
        return False

if __name__ == "__main__":
    check_istat_sdmx_status()
    