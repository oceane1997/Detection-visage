import streamlit as st
import cv2
import numpy as np
import os, time, io, zipfile
from PIL import Image

# Charge le classificateur frontal par défaut (portables Win/Mac/Linux)
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

def read_image_to_bgr(file):
    """Lit un fichier image (uploader/camera_input) → numpy BGR pour OpenCV."""
    img = Image.open(file).convert("RGB")   # PIL lit en RGB
    arr = np.array(img)                     # (H,W,3) RGB
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return bgr

def detect_faces_bgr(bgr, scaleFactor=1.3, minNeighbors=5):
    """Détecte des visages avec Viola–Jones sur une image BGR."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors)
    return faces  # liste de (x,y,w,h)

def annotate_rects(bgr, faces, color_bgr=(0,255,0), thickness=2):
    """Dessine les rectangles autour des visages (copie l’image)."""
    out = bgr.copy()
    for (x,y,w,h) in faces:
        cv2.rectangle(out, (x,y), (x+w, y+h), color_bgr, thickness)
    return out

def save_cropped_faces(bgr, faces, prefix="face"):
    """Sauve chaque visage recadré en PNG dans ./outputs et renvoie la liste des chemins."""
    os.makedirs("outputs", exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    saved = []
    for i, (x,y,w,h) in enumerate(faces, start=1):
        crop = bgr[y:y+h, x:x+w]
        path = os.path.join("outputs", f"{prefix}_{ts}_{i}.png")
        cv2.imwrite(path, crop)
        saved.append(path)
    return saved

def zip_bytes(paths):
    """Crée un ZIP en mémoire (BytesIO) à partir d’une liste de chemins."""
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, "w", zipfile.ZIP_DEFLATED) as z:
        for p in paths:
            z.write(p, arcname=os.path.basename(p))
    mem.seek(0)
    return mem

def hex_to_bgr(hex_color: str) -> tuple[int, int, int]:
    """#RRGGBB -> (B, G, R) pour OpenCV"""
    h = hex_color.lstrip("#")
    r = int(h[0:2], 16)
    g = int(h[2:4], 16)
    b = int(h[4:6], 16)
    return (b, g, r)



# Config de la page (titre de l’onglet + emoji)
st.set_page_config(page_title="Détection de visages (Viola–Jones)", page_icon="📸")

# Titre visible sur la page
st.title("📸 Détection de visages — Algorithme de Viola–Jones")

# Bloc d'instructions utilisateur
st.markdown("""
### Comment utiliser cette application
1. **Autorisez l'accès à la caméra** lorsque Streamlit le demande *(ou préparez une photo à charger)*.
2. Vous pourrez **prendre une photo** avec la webcam ou **charger une image** depuis votre ordinateur.
3. Ensuite, vous réglerez :
   - la **couleur** des rectangles,
   - la **sensibilité** (*scaleFactor*),
   - le **voisinage minimal** (*minNeighbors*).
4. Cliquez sur **Détecter** pour lancer l'algorithme et sur **Enregistrer** pour sauvegarder l'image annotée.

> Astuce : placez-vous face caméra, avec un éclairage homogène. Les lunettes/masques peuvent réduire la détection.
""")




st.markdown("---")
st.subheader("Détection sur une image et sauvegarde des visages")

tab_cam, tab_file = st.tabs(["📷 Prendre une photo", "📁 Importer une image"])

with tab_cam:
    cam_img = st.camera_input("Prenez une photo (autorisez la webcam)")
with tab_file:
    up_img = st.file_uploader("Choisissez une image (jpg/png)", type=["jpg","jpeg","png"])

img_file = cam_img or up_img

# Choix de couleur (par défaut: vert)
color_hex = st.color_picker("Couleur des rectangles", "#00FF00")
color_bgr = hex_to_bgr(color_hex)

# Nombre de voisins requis pour valider une détection (plus haut = plus strict)
min_neighbors = st.slider("minNeighbors", min_value=1, max_value=15, value=5, step=1,
                          help="Plus élevé ⇒ moins de faux positifs, mais risque de rater des visages.")

# Échelle du pyramidage (plus proche de 1.0 = plus sensible mais plus lent)
scale_factor = st.slider(
    "scaleFactor",
    min_value=1.05, max_value=1.50, value=1.30, step=0.01,
    help="Plus proche de 1 ⇒ plus de niveaux d’échelle (détection plus fine mais plus lente)."
)


if img_file is None:
    st.warning("Prenez une photo ou importez une image pour lancer la détection.")
else:
    # Lecture et détection (paramètres par défaut pour l’instant)
    bgr = read_image_to_bgr(img_file)
    faces = detect_faces_bgr(bgr, scaleFactor=scale_factor, minNeighbors=min_neighbors)



    # Annotation (couleur verte par défaut)
    annotated = annotate_rects(bgr, faces, color_bgr=color_bgr, thickness=2)

    # Convertir BGR → RGB pour l’affichage Streamlit
    rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    # Affichage
    col1, col2 = st.columns(2)
    with col1:
        st.image(rgb, caption=f"Image annotée — visages détectés : {len(faces)}", use_container_width=True)
    with col2:
        if len(faces) == 0:
            st.info("Aucun visage détecté. Essayez une autre image ou un meilleur éclairage.")
        else:
            # Sauvegarde locale (serveur) des visages recadrés
            saved_paths = save_cropped_faces(bgr, faces, prefix="face")
            st.success(f"{len(saved_paths)} visage(s) sauvegardé(s) dans le dossier ./outputs")

            # Proposer un téléchargement en ZIP vers l’appareil de l’utilisateur
            zipbuf = zip_bytes(saved_paths)
            st.download_button(
                "⬇️ Télécharger les visages recadrés (ZIP)",
                data=zipbuf,
                file_name="faces_detected.zip",
                mime="application/zip"
            )

            # Option : proposer aussi de télécharger l'image annotée
            _, buf_png = cv2.imencode(".png", annotated)
            st.download_button(
                "⬇️ Télécharger l'image annotée (PNG)",
                data=buf_png.tobytes(),
                file_name="annotated.png",
                mime="image/png"
            )


