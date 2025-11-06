import json
import math
import requests
import random
import re
import os
from datetime import datetime

# =============================================================================
# UTILITAIRES
# =============================================================================

def sliding_window(data, window_size, overlap):
    """
    Génère des fenêtres glissantes de taille 'window_size'
    avec un chevauchement de 'overlap' éléments.
    """
    step = window_size - overlap
    for i in range(0, len(data), step):
        yield data[i:i + window_size]

def minimal_profile(profile):
    """
    Garde seulement les champs essentiels pour limiter les tokens.
    Tronque les textes trop longs (bio).
    """
    keys = ['username', 'full_name', 'bio', 'location', 'source']
    filtered = {k: profile.get(k) for k in keys if profile.get(k) is not None}

    # Tronque les bios trop longues pour éviter dépassement de tokens
    if 'bio' in filtered and len(filtered['bio']) > 400:
        filtered['bio'] = filtered['bio'][:400] + "..."

    return filtered

import json
import re
import ast

def extract_json_from_text(text):
    """
    Extrait le premier bloc JSON valide (objet ou liste) d'une réponse LLM.
    Gère les cas où le texte contient du texte ou du formatage avant/après.
    """
    # Cherche le premier [ ou { et le dernier ] ou }
    match = re.search(r'(\{.*\}|\[.*\])', text, re.DOTALL)
    if not match:
        raise ValueError("Aucun JSON détecté dans la réponse du LLM.")

    json_str = match.group(1).strip()

    # Essaye d’abord le parser JSON standard
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        # Si c’est un format Python-like, on essaie ast.literal_eval
        try:
            return ast.literal_eval(json_str)
        except Exception as e:
            raise ValueError(f"Impossible de parser le JSON extrait : {e}\nTexte brut : {json_str[:300]}...")

# =============================================================================
# APPEL API LLM (OpenRouter)
# =============================================================================

def call_llm_api(prompt, api_key, model_name="deepseek/deepseek-r1-0528-qwen3-8b:free"):
    """
    Envoie un prompt au modèle LLM via l'API OpenRouter.
    Force une sortie strictement JSON.
    """
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    messages = [
        {"role": "system", "content": "Répondez STRICTEMENT avec un JSON valide (liste d'objets). Aucune explication, aucun texte, aucune balise markdown."},
        {"role": "user", "content": prompt}
    ]
    payload = {
        "model": model_name,
        "messages": messages,
        "max_tokens": 1200,
        "temperature": 0.0
    }

    response = requests.post(url, json=payload, headers=headers, timeout=120)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Erreur API : {response.status_code} - {response.text}")

# =============================================================================
# CONSTRUCTION DU PROMPT
# =============================================================================

def build_prompt(profiles_batch):
    """
    Construit un prompt robuste pour garantir une réponse strictement en JSON valide.
    """
    instruction = (
        "Tu es un assistant de correspondance de profils provenant de différentes plateformes "
        "(LinkedIn, Twitter, GitHub).\n\n"
        "Ta tâche : identifier uniquement les profils qui semblent appartenir à la même personne "
        "mais provenant de plateformes différentes.\n\n"
        "⚠️ RÈGLES DE SORTIE STRICTES :\n"
        "- Ne retourne **aucun texte, explication ou commentaire** avant ou après le JSON.\n"
        "- Ne mets **aucune balise markdown** (` ``` `, etc.).\n"
        "- La réponse doit être **un JSON valide**, parsable directement par `json.loads()`.\n"
        "- Si aucune correspondance n’est trouvée, réponds **exactement** : []\n\n"
        "Format JSON attendu :\n"
        "[\n"
        "  {\n"
        "    \"profile1\": {\"username\": \"...\", \"source\": \"LinkedIn\"},\n"
        "    \"profile2\": {\"username\": \"...\", \"source\": \"GitHub\"},\n"
        "    \"score\": 0.85\n"
        "  },\n"
        "  {\n"
        "    \"profile1\": {\"username\": \"...\", \"source\": \"Twitter\"},\n"
        "    \"profile2\": {\"username\": \"...\", \"source\": \"LinkedIn\"},\n"
        "    \"score\": 0.90\n"
        "  }\n"
        "]\n\n"
        "Assure-toi que tous les objets respectent cette structure, sans texte additionnel.\n"
    )

    # Conversion du batch en JSON lisible par le modèle
    profiles_json = json.dumps(profiles_batch, ensure_ascii=False, indent=2)
    
    # Construction finale du prompt
    prompt = f"{instruction}\nProfils à comparer :\n{profiles_json}\n\nRéponse :"
    
    return prompt

# =============================================================================
# TRAITEMENT PRINCIPAL DES PROFILS
# =============================================================================

def process_profiles_with_llm(profiles, api_key, batch_size=70, overlap=20, output_dir="llm_outputs_deepseek"):
    os.makedirs(output_dir, exist_ok=True)
    all_matches = []

    # Mélange aléatoire pour diversifier les plateformes dans chaque batch
    random.seed(42)
    random.shuffle(profiles)

    total_batches = math.ceil(len(profiles) / (batch_size - overlap))
    print(f"[INFO] Début traitement de {len(profiles)} profils en {total_batches} fenêtres glissantes.")

    for idx, batch in enumerate(sliding_window(profiles, batch_size, overlap)):
        batch_number = idx + 1
        print(f"[{datetime.utcnow().isoformat()}] Fenêtre {batch_number}/{total_batches} ({len(batch)} profils)")

        minimal_batch = [minimal_profile(p) for p in batch]
        prompt = build_prompt(minimal_batch)

        try:
            result = call_llm_api(prompt, api_key)
            llm_response_text = result['choices'][0]['message']['content']
        except Exception as e:
            err_file = os.path.join(output_dir, f"batch_{batch_number}_api_error.txt")
            with open(err_file, "w", encoding="utf-8") as f:
                f.write(str(e))
            print(f"[ERREUR API] Fenêtre {batch_number} : {e}")
            continue

        raw_file = os.path.join(output_dir, f"batch_{batch_number}_raw.txt")
        with open(raw_file, "w", encoding="utf-8") as f:
            f.write(llm_response_text)

        try:
            parsed = extract_json_from_text(llm_response_text)
            if isinstance(parsed, list):
                all_matches.extend(parsed)
            else:
                all_matches.append(parsed)
            out_file = os.path.join(output_dir, f"batch_{batch_number}_matches.json")
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(parsed, f, indent=2, ensure_ascii=False)
            print(f"[OK] Fenêtre {batch_number} : {len(parsed) if isinstance(parsed, list) else 1} correspondances.")
        except json.JSONDecodeError:
            print(f"[ERREUR JSON] Fenêtre {batch_number} non parsée, résultat brut sauvegardé.")
            continue
        except ValueError:
            print(f"[WARN] Aucun JSON détecté dans la fenêtre {batch_number}. Sauvegarde brute...")
            raw_file = os.path.join(output_dir, f"batch_{batch_number:03d}_raw.txt")
            with open(raw_file, "w", encoding="utf-8") as f:
                f.write(llm_response_text)
            continue

    return all_matches

# =============================================================================
# CHARGEMENT DES FICHIERS
# =============================================================================

def load_json_with_source(filepath, source_name, limit=None):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if limit:
        data = data[:limit]
    for profile in data:
        profile['source'] = source_name
    return data

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # ⚠️ Insère ta clé API ici
    API_KEY = "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"

    # --- Chargement des fichiers ---
    linkedin_profiles = load_json_with_source("linkedin_profiles_normalized.json", "LinkedIn")
    twitter_profiles = load_json_with_source("twitter_data_cleaned.json", "Twitter")
    github_profiles = load_json_with_source("github_cleaned.json", "GitHub")

    all_profiles = linkedin_profiles + twitter_profiles + github_profiles
    print(f"[INFO] Nombre total de profils chargés : {len(all_profiles)}")

    # --- Lancement du traitement ---
    matches = process_profiles_with_llm(
        all_profiles,
        api_key=API_KEY,
        batch_size=70,      # Taille de la fenêtre
        overlap=20,         # Chevauchement entre fenêtres
        output_dir="llm_outputs_deepseek"
    )

    # --- Sauvegarde globale ---
    with open("matched_profiles.json", "w", encoding="utf-8") as f:
        json.dump(matches, f, indent=2, ensure_ascii=False)

    print(f"[FIN] Matching terminé. {len(matches)} correspondances totales extraites.")
