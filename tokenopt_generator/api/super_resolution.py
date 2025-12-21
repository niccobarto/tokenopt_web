import os
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
        out_path = os.path.join(out_dir, "output.png")

        # Scriviamo il file di input
        with open(in_path, "wb") as f:
            f.write(input_bytes)

        # Costruiamo il comando
        cmd=[part.format(in_path=in_path, out_path=out_path, out_dir=out_dir) for part in sr_cli_cmd]

        # Lancio la CLI e raccolgo stdout/stderr per debug
        completed = subprocess.run(cmd,
                                   capture_output=True,
                                   text=True)

        if completed.returncode != 0:
            raise RuntimeError(
                "Real-ESRGAN non è andato a buon fine.\n"
                f"Command: {cmd}\n\n"
                f"STDOUT:\n{completed.stdout}\n\n"
                f"STDERR:\n{completed.stderr}"
            )

        if not os.path.exists(out_path):
            raise RuntimeError(
                "Real-ESRGAN è terminato senza creare l'output atteso.\n"
                f"Expected output: {out_path}\n"
                f"Command: {cmd}\n\n"
                f"STDOUT:\n{completed.stdout}\n\n"
                f"STDERR:\n{completed.stderr}"
            )
        with open(out_path, "rb") as f:
            return f.read()