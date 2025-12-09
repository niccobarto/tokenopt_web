#!/usr/bin/env python

import argparse
from pathlib import Path

from PIL import Image
from rembg import remove, new_session

def remove_background(
    input_path: Path,
    output_path: Path | None = None,
    model_name: str = "isnet-general-use",
) -> Path:
    """
    Rimuove lo sfondo da un'immagine usando rembg e salva un PNG con alpha.

    Args:
        input_path: percorso dell'immagine di input.
        output_path: percorso di output (se None -> stesso nome con suffix '-nobg.png').
        model_name: nome del modello rembg (es. 'u2net', 'isnet-general-use', ...).

    Returns:
        Path del file di output salvato.
    """
    if output_path is None:
        output_path = input_path.with_suffix("")  # togli estensione
        output_path = output_path.with_name(output_path.name + "-nobg.png")

    # Crea una sessione del modello una sola volta (più efficiente se chiami più volte)
    session = new_session(model_name)

    # Apri immagine e converti in RGBA (serve il canale alpha per la trasparenza)
    with Image.open(input_path) as img:
        img = img.convert("RGBA")
        # rembg restituisce un oggetto PIL.Image con lo sfondo già rimosso
        result = remove(img, session=session)

    # Salva come PNG per mantenere l'alpha
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(output_path)

    print(f"[OK] Salvato file senza sfondo in: {output_path}")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rimuovi lo sfondo da una immagine usando rembg."
    )
    parser.add_argument(
        "input",
        type=str,
        help="Percorso dell'immagine di input (JPG/PNG, ecc.).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Percorso di output (default: stesso nome con suffix '-nobg.png').",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="isnet-general-use",
        help="Nome modello rembg (es. 'u2net', 'isnet-general-use', 'u2net_human_seg', ...).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    in_path = Path(args.input)
    if not in_path.is_file():
        raise FileNotFoundError(f"Input non trovato: {in_path}")

    out_path = Path(args.output) if args.output is not None else None

    remove_background(
        input_path=in_path,
        output_path=out_path,
        model_name=args.model,
    )


if __name__ == "__main__":
    main()
