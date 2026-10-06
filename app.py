import json

import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

st.set_page_config(page_title="Détecteur de frelon", page_icon="🐝")

# ---------------------------------------------------------------------------
# 1. Configuration : les mêmes réglages qu'à l'entraînement
# ---------------------------------------------------------------------------
with open("config.json", encoding="utf-8") as f:
    CONFIG = json.load(f)
CLASSES = CONFIG["classes"]            # l'ordre des classes = l'ordre des sorties du modèle


# ---------------------------------------------------------------------------
# 2. Chargement du modèle (une seule fois grâce au cache)
# ---------------------------------------------------------------------------
@st.cache_resource
def charger_modele():
    modele = models.efficientnet_b2(weights=None)        # structure seule
    modele.classifier = nn.Sequential(                    # notre tête, identique à l'entraînement
        nn.Linear(1408, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, len(CLASSES)),
    )
    poids = torch.load("modele_frelon.pt", map_location="cpu")
    poids = {nom: (t.float() if t.is_floating_point() else t) for nom, t in poids.items()}
    modele.load_state_dict(poids)                         # nos poids entraînés (float16 -> float32)
    modele.eval()                                         # mode évaluation : Dropout désactivé
    return modele


modele = charger_modele()

transfo = transforms.Compose([
    transforms.Resize((CONFIG["taille"], CONFIG["taille"])),
    transforms.ToTensor(),
    transforms.Normalize(CONFIG["moyenne"], CONFIG["ecart_type"]),
])


# ---------------------------------------------------------------------------
# 3. Prédiction
# ---------------------------------------------------------------------------
def predire(image):
    x = transfo(image).unsqueeze(0)                       # [3, 288, 288] -> [1, 3, 288, 288]
    with torch.no_grad():
        probas = torch.softmax(modele(x), dim=1)[0]      # 6 scores -> 6 probabilités (somme = 1)
    return probas.numpy()


# ---------------------------------------------------------------------------
# 4. Interface
# ---------------------------------------------------------------------------
st.title("🐝 Détecteur de frelon asiatique")
st.write("Envoyez la photo d'un insecte : le modèle indique de quelle espèce il s'agit.")

fichier = st.file_uploader("Choisir une photo", type=["jpg", "jpeg", "png"])

if fichier is not None:
    image = Image.open(fichier).convert("RGB")
    st.image(image, width=400)

    probas = predire(image)
    meilleure = probas.argmax()
    st.subheader(f"Prédiction : {CONFIG['noms'][CLASSES[meilleure]]}")
    st.write(f"Confiance : {probas[meilleure]:.0%}")

    tableau = pd.Series({CONFIG["noms"][c]: float(p) for c, p in zip(CLASSES, probas)},
                        name="Probabilité")
    st.bar_chart(tableau)
