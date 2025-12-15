from PIL import Image
import io
# cache di sessioni per modello (valida per tutto il processo Celery)
_SESSIONS: dict[str, object] = {}

def remove_background(input_bytes:bytes,model_selected:str)->bytes:
    from rembg import remove,new_session
    """
    Rimuove lo sfondo usando rembg e il modello selezionato.
    Ritorna PNG con canale alpha (bytes).
    """

    if model_selected not in _SESSIONS:
        _SESSIONS[model_selected] = new_session(model_selected)
    session = _SESSIONS[model_selected]
     #leggo immagine da bytes
    input_image=Image.open(io.BytesIO(input_bytes)).convert("RGBA")
    output_image = remove(input_image,session=session) #rembg rimuove sfondo, ritorna PIL Image se ricevuto PIL Image

    #ritorna bytes PNG
    if isinstance(output_image, bytes):
        return output_image

    output_bytes_io = io.BytesIO()
    output_image.save(output_bytes_io, format="PNG")
    return output_bytes_io.getvalue()