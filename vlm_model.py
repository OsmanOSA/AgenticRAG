def vll_generate_description_image(image_path: str,
                                   model: str = "llava") -> str:
    """Génère une description factuelle d'une image via un VLM open-source local.

    Utilise Ollama comme backend (modèles compatibles : llava, moondream,
    minicpm-v, etc.). L'image est lue depuis le disque et envoyée au modèle
    en bytes.

    Parameters
    ----------
    image_path : str
        Chemin vers l'image à décrire.
    model : str, optional
        Nom du modèle Ollama vision à utiliser (défaut : "llava").

    Returns
    -------
    str
        Description factuelle de l'image, ou chaîne vide si l'image
        est introuvable ou si le modèle échoue.
    """
    try:
        # Résolution du chemin : essaie tel quel, puis depuis le CWD
        img = Path(image_path)
        if not img.exists():
            img = Path.cwd() / image_path
        if not img.exists():
            print(f"[VLM] ⚠ Image introuvable : {image_path}")
            return ""

        print(f"[VLM] Traitement : {img.name}")
        with open(img, "rb") as f:
            image_bytes = f.read()

        response = ollama.chat(
            model=model,
            messages=[{
                "role": "user",
                "content": (
                    "Décris factuellement cette image en 2 ou 3 phrases. "
                    "Concentre-toi sur le contenu technique ou informatif visible "
                    "(schéma, tableau, graphique, formule, diagramme...). "
                    "Réponds en français."
                ),
                "images": [image_bytes],
            }],
        )

        description = response["message"]["content"].strip()
        print(f"[VLM] ✓ Description générée ({len(description)} chars)")
        return description

    except Exception as e:
        print(f"[VLM] ✗ Échec pour {image_path} : {e}")
        return ""


def enrich_with_image_descriptions(documents: List[Document],
                                   model: str = "llava") -> List[Document]:
    """Enrichit les documents en insérant des descriptions VLM sous chaque image.

    Parcourt chaque ``Document``, détecte les liens Markdown de la forme
    ``![alt](path)``, génère une description via ``vll_generate_description_image``
    et l'insère juste après sous la forme d'un blockquote italique.

    Avant  : ``![](./images/page_2_img_0.png)``
    Après  : ``![](./images/page_2_img_0.png)``
             ``> *Schéma montrant l'interaction entre les modules MLOps et le Cloud.*``

    Parameters
    ----------
    documents : List[Document]
        Documents issus de ``pdf_extractor_pymupdf4llm``.
    model : str, optional
        Modèle Ollama vision (défaut : "llava").

    Returns
    -------
    List[Document]
        Documents enrichis avec descriptions d'images insérées.
    """
    IMAGE_RE = re.compile(r'!\[([^\]]*)\]\(([^)]+)\)')

    def _replace(match: re.Match) -> str:
        alt, path = match.group(1), match.group(2)
        description = vll_generate_description_image(path, model=model)
        if description:
            return f"![{alt}]({path})\n> *{description}*"
        return match.group(0)

    try:
        # Compte le nombre total d'images à traiter
        total_images = sum(
            len(IMAGE_RE.findall(doc.page_content)) for doc in documents
        )
        print(f"[VLM] {total_images} image(s) détectée(s) sur {len(documents)} page(s)")

        enriched = []
        for doc in documents:
            new_content = IMAGE_RE.sub(_replace, doc.page_content)
            enriched.append(Document(
                page_content=new_content,
                metadata=doc.metadata,
            ))

        print(f"[VLM] Enrichissement terminé sur {len(enriched)} document(s)")
        return enriched

    except Exception as e:
        raise AgenticRagException(e, sys)