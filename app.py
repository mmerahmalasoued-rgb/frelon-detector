import json

import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

st.set_page_config(page_title="Frelon asiatique ou pas ?", page_icon="🐝", layout="wide")

# ---------------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------------
with open("config.json", encoding="utf-8") as f:
    CONFIG = json.load(f)
CLASSES = CONFIG["classes"]
NOMS = CONFIG["noms"]
SEUIL_CONFIANCE = 0.60     # en dessous, on prévient que le modèle hésite
EQUIPE = "Prénom NOM & Prénom NOM"   # à remplacer par les noms du binôme

# Conseil affiché selon l'espèce prédite : (fonction d'affichage Streamlit, texte)
CONSEILS = {
    "frelon_asiatique": (st.error, "⚠️ **Suspicion de frelon asiatique**, espèce invasive qui s'attaque "
                                   "aux abeilles. Ne vous approchez pas d'un éventuel nid. Faites confirmer "
                                   "l'identification, puis signalez-le à votre mairie ou au dispositif de "
                                   "signalement de votre département."),
    "frelon_europeen": (st.info, "🟤 **Frelon européen** : espèce indigène, utile à l'écosystème. "
                                 "Plutôt pacifique loin de son nid : inutile de le détruire."),
    "guepe": (st.warning, "🟡 **Guêpe** : elle peut piquer si elle se sent menacée. "
                          "Restez calme et évitez les grands gestes."),
    "abeille": (st.success, "🐝 **Abeille domestique** : pollinisateur précieux. Aucune action nécessaire."),
    "bourdon": (st.success, "🐝 **Bourdon** : pollinisateur pacifique, très peu agressif."),
    "volucelle": (st.success, "🪰 **Volucelle zonée** : une mouche totalement inoffensive qui *imite* "
                              "le frelon pour se protéger de ses prédateurs."),
}


# ---------------------------------------------------------------------------
# 2. Modèle (chargé une seule fois grâce au cache)
# ---------------------------------------------------------------------------
@st.cache_resource
def charger_modele():
    modele = models.efficientnet_b2(weights=None)          # structure seule
    modele.classifier = nn.Sequential(                      # notre tête, identique à l'entraînement
        nn.Linear(1408, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, len(CLASSES)),
    )
    poids = torch.load("modele_frelon.pt", map_location="cpu")
    poids = {nom: (t.float() if t.is_floating_point() else t) for nom, t in poids.items()}
    modele.load_state_dict(poids)                           # nos poids entraînés (float16 -> float32)
    modele.eval()                                           # mode évaluation : Dropout désactivé
    return modele


modele = charger_modele()

transfo = transforms.Compose([
    transforms.Resize((CONFIG["taille"], CONFIG["taille"])),
    transforms.ToTensor(),
    transforms.Normalize(CONFIG["moyenne"], CONFIG["ecart_type"]),
])


def predire(image):
    x = transfo(image).unsqueeze(0)                         # [3, 288, 288] -> [1, 3, 288, 288]
    with torch.no_grad():
        probas = torch.softmax(modele(x), dim=1)[0]        # 6 scores -> 6 probabilités (somme = 1)
    return probas.numpy()


# ---------------------------------------------------------------------------
# 3. Panneau latéral : présentation du projet
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("À propos du projet")
    st.metric("Précision sur le jeu de test", f"{CONFIG['precision_test']:.1%}")
    st.markdown(
        "**Modèle** : EfficientNet-B2 pré-entraîné sur ImageNet, fine-tuné en PyTorch  \n"
        "**Données** : ≈ 4 800 photos d'observations validées en France "
        "(iNaturalist, licences Creative Commons)  \n"
        "**Espèces** : frelon asiatique, frelon européen, guêpe, abeille, bourdon "
        "et volucelle zonée, une mouche qui imite le frelon"
    )
    st.warning("Outil d'aide à l'identification : il peut se tromper et ne remplace pas l'avis d'un expert.")
    st.caption(f"Projet Advanced Practical Machine Learning · aivancity  \n{EQUIPE}")


# ---------------------------------------------------------------------------
# 4. En-tête
# ---------------------------------------------------------------------------
st.title("🐝 Frelon asiatique ou pas ?")
st.write(
    "Le frelon asiatique est une espèce invasive qui décime les ruches en France. "
    "Il est souvent confondu avec le frelon européen, les guêpes ou même une mouche qui l'imite. "
    "Fournissez la photo d'un insecte : le modèle identifie l'espèce et vous conseille."
)

# ---------------------------------------------------------------------------
# 5. Choix de l'image
# ---------------------------------------------------------------------------
source = st.radio("Source de l'image :",
                  ["📁 Envoyer une photo", "🖼️ Essayer un exemple", "📸 Prendre une photo"],
                  horizontal=True)

image, credit = None, None

if source == "📁 Envoyer une photo":
    fichier = st.file_uploader("Choisissez une photo d'insecte", type=["jpg", "jpeg", "png"])
    if fichier is not None:
        image = Image.open(fichier).convert("RGB")

elif source == "🖼️ Essayer un exemple":
    st.caption("Photos du jeu de test : le modèle ne les a jamais vues pendant l'entraînement.")
    colonnes = st.columns(len(CLASSES))
    for numero, (colonne, classe) in enumerate(zip(colonnes, CLASSES), start=1):
        with colonne:
            st.image(f"exemples/{classe}.jpg", width=140)
            if st.button(f"Exemple {numero}", key=classe):
                st.session_state["exemple"] = classe      # on mémorise le choix entre deux relances
    if "exemple" in st.session_state:
        classe = st.session_state["exemple"]
        image = Image.open(f"exemples/{classe}.jpg").convert("RGB")
        credits = pd.read_csv("exemples/credits.csv")
        ligne = credits[credits["fichier"] == f"{classe}.jpg"].iloc[0]
        credit = f"Photo : {ligne['auteur']} · licence {ligne['licence']} · via iNaturalist"

else:
    photo = st.camera_input("Prenez l'insecte en photo")
    if photo is not None:
        image = Image.open(photo).convert("RGB")

# ---------------------------------------------------------------------------
# 6. Résultat
# ---------------------------------------------------------------------------
if image is not None:
    probas = predire(image)
    ordre = probas.argsort()[::-1]                 # indices des classes, de la plus probable à la moins probable
    meilleure = CLASSES[ordre[0]]
    confiance = float(probas[ordre[0]])
    incertain = confiance < SEUIL_CONFIANCE

    st.divider()
    col_image, col_resultat = st.columns([1, 1.3])

    with col_image:
        st.image(image, width=380)
        if credit:
            st.caption(credit)

    with col_resultat:
        st.subheader("Résultat")
        st.markdown(f"## {NOMS[meilleure]}{' ?' if incertain else ''}")
        st.progress(confiance, text=f"Confiance du modèle : {confiance:.0%}")

        if incertain:
            seconde = CLASSES[ordre[1]]            # deuxième espèce la plus probable
            st.warning(f"🤔 **Identification incertaine** : le modèle hésite entre "
                       f"**{NOMS[meilleure]}** ({confiance:.0%}) et "
                       f"**{NOMS[seconde]}** ({probas[ordre[1]]:.0%}). "
                       "Essayez une photo plus nette, où l'insecte occupe une plus grande partie de l'image.")
        else:
            afficher, texte = CONSEILS[meilleure]
            afficher(texte)

        st.markdown("**Détail des probabilités**")
        for i in ordre:
            st.progress(float(probas[i]), text=f"{NOMS[CLASSES[i]]} : {probas[i]:.1%}")

# ---------------------------------------------------------------------------
# 7. Explications et limites
# ---------------------------------------------------------------------------
with st.expander("🔍 Comment fonctionne le modèle ? Quelles sont ses limites ?"):
    st.markdown(
        "**Fonctionnement.** Le modèle part d'un réseau EfficientNet-B2 déjà entraîné sur 1,2 million "
        "d'images (ImageNet). Nous avons remplacé sa dernière couche par notre propre tête de "
        "classification (1408 → 256 → 6), puis ré-entraîné ses derniers blocs sur nos photos "
        "d'insectes : c'est le *fine-tuning*.\n\n"
        "**Limites connues.**\n"
        "- Le modèle ne connaît que 6 espèces : tout autre insecte sera forcément classé dans l'une d'elles.\n"
        "- Il se trompe plus souvent quand l'insecte est petit, flou ou caché dans le décor.\n"
        "- La volucelle imite le frelon européen : le modèle se fait parfois piéger, comme un humain.\n"
        "- Environ 1 alerte « frelon asiatique » sur 4 est une fausse alerte : une confirmation reste nécessaire."
    )
