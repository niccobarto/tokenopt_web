import os
import glob
import subprocess
import tempfile


def run_realesgan(input_bytes: bytes, sr_cli_cmd: list[str]) -> bytes:
    """
    Esegue la super-risoluzione tramite Real-ESRGAN come black-box esterna
    richiamata via linea di comando.

    Parametri
    ----------
    input_bytes : bytes
        Contenuto binario dell'immagine di input (PNG/JPEG).
        L'immagine viene fornita in memoria, non come file persistente.

    sr_cli_cmd : list[str]
        Comando CLI (template) definito in settings.py.
        Deve contenere i placeholder:
            - {in_path}  : path del file di input
            - {out_dir}  : directory di output

        Esempio:
        [
            "python", "inference_realesrgan.py",
            "-i", "{in_path}",
            "-o", "{out_dir}",
            "-n", "realesrgan-x4plus"
        ]

    Ritorna
    -------
    bytes
        Contenuto binario dell'immagine super-risoluta prodotta dal modello.

    Solleva
    -------
    RuntimeError
        Se l'esecuzione della CLI fallisce o non produce output.
    """

    # Crea una directory temporanea che verrà automaticamente eliminata
    # al termine del blocco 'with'
    with tempfile.TemporaryDirectory() as tmpdir:

        # Percorso del file di input per la SR
        in_path = os.path.join(tmpdir, "input.png")

        # Directory in cui il tool SR scriverà l'immagine di output
        out_dir = os.path.join(tmpdir, "out")
        os.makedirs(out_dir, exist_ok=True)

        # Scrive l'immagine di input (bytes) su file temporaneo
        # Questo è necessario perché Real-ESRGAN lavora su file
        with open(in_path, "wb") as f:
            f.write(input_bytes)

        # Costruisce il comando CLI sostituendo i placeholder
        # con i path reali della directory temporanea
        cmd = [
            arg.format(in_path=in_path, out_dir=out_dir)
            for arg in sr_cli_cmd
        ]

        # Esegue il comando come processo esterno
        # stdout e stderr vengono catturati per debugging
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True
        )

        # Se il processo termina con codice != 0,
        # la super-risoluzione è fallita
        if completed.returncode != 0:
            raise RuntimeError(
                "Real-ESRGAN CLI failed.\n"
                f"Command: {cmd}\n\n"
                f"STDOUT:\n{completed.stdout}\n\n"
                f"STDERR:\n{completed.stderr}"
            )

        # Real-ESRGAN non garantisce un nome fisso del file di output.
        # Cerchiamo quindi qualsiasi immagine prodotta nella directory out/
        candidates = []
        candidates.extend(glob.glob(os.path.join(out_dir, "*.png")))
        candidates.extend(glob.glob(os.path.join(out_dir, "*.jpg")))
        candidates.extend(glob.glob(os.path.join(out_dir, "*.jpeg")))

        # Se non troviamo alcun file, consideriamo il job fallito
        if not candidates:
            raise RuntimeError(
                "Real-ESRGAN non ha prodotto alcun file di output.\n"
                f"Command: {cmd}\n\n"
                f"STDOUT:\n{completed.stdout}\n\n"
                f"STDERR:\n{completed.stderr}"
            )

        # Se vengono prodotti più file, scegliamo quello più recente
        # (comportamento corretto per una SR su singola immagine)
        out_path = max(candidates, key=os.path.getmtime)

        # Legge il file di output e lo restituisce come bytes
        # Da qui in poi, il resto del sistema (Django/Celery)
        # non ha più bisogno di sapere come è stata fatta la SR
        with open(out_path, "rb") as f:
            return f.read()
