from PIL import Image
import io


def remove_background(input_bytes:bytes)->bytes:
    from rembg import remove
    """Ritorna PNG con trasparenza (alpha)"""

    input_image=Image.open(io.BytesIO(input_bytes)).convert("RGBA")

    output_image = remove(input_image) #rembg rimuove sfondo, ritorna PIL Image se ricevuto PIL Image

    #ritorna bytes PNG
    if isinstance(output_image, bytes):
        return output_image

    output_bytes_io = io.BytesIO()
    output_image.save(output_bytes_io, format="PNG")
    return output_bytes_io.getvalue()