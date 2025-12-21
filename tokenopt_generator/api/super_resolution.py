import os
import glob
import subprocess
import tempfile


def run_realesgan(input_bytes: bytes, sr_cli_cmd: list[str]) -> bytes:
    """
    Esegue la super-risoluzione usando Real-ESRGAN CLI.
    Scrive l'output in un comando temporaneo, lancia la CLI e legge il file output prodotto
    """

    if not sr_cli_cmd:
        raise RuntimeError("Comando CLI per Real-ESRGAN non fornito")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Prepariamo i path e le cartelle
        in_path=os.path.join(tmpdir, "input.png")
        out_dir=os.path.join(tmpdir, "out")
        os.makedirs(out_dir, exist_ok=True)

        # Scriviamo il file di input
        with open(in_path, "wb") as f:
            f.write(input_bytes)

        # Costruiamo il comando
        cmd=[part.format(in_path=in_path, out_dir=out_dir) for part in sr_cli_cmd]

        # Lancio la CLI e raccolgo stdout/stderr per debug
        completed = subprocess.run(cmd, capture_output=True, text=True)

        if completed.returncode != 0:
            raise RuntimeError(
                "Real-ESRGAN non è andato a buon fine.\n"
                f"Command: {cmd}\n\n"
                f"STDOUT:\n{completed.stdout}\n\n"
                f"STDERR:\n{completed.stderr}"            )

        #cerco il file di output
        candidates:list[str]=[]

        candidates.extend(glob.glob(os.path.join(out_dir, "*.png")))
        candidates.extend(glob.glob(os.path.join(out_dir, "*.jpg")))
        candidates.extend(glob.glob(os.path.join(out_dir, "*.jpeg")))

        if not candidates:
            raise RuntimeError(
                "Real-ESRGAN non ha creato immagini di output.\n"
                f"Command: {cmd}\n\n"
                f"STDOUT:\n{completed.stdout}\n\n"
                f"STDERR:\n{completed.stderr}"
            )
        # Se ci sono più file, prendo il più recente (in genere è l'unico)
        out_path = max(candidates, key=os.path.getmtime)

        # Leggo il risultato e lo restituisco come bytes
        with open(out_path, "rb") as f:
            return f.read()