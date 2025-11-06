import json
import math
import requests
import random
import re
import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()   

# --- Utilities --------------------------------------------------------------
def chunk_list(data, batch_size):
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]

def minimal_profile(profile):
    """
    Garde seulement les champs essentiels pour limiter les tokens.
    Ajoute/modifie selon ta structure de données.
    """
    keys = ['username', 'full_name', 'bio', 'location', 'source']
    return {k: profile.get(k) for k in keys if profile.get(k) is not None}

def extract_json_from_text(text):
    """
    Essaye d'extraire le premier JSON valide trouvé dans le texte.
    Gère cas courant : triple backticks, code fences, ou JSON inline.
    Retourne l'objet Python ou raise JSONDecodeError si échec.
    """
    text = text.strip()

    # 1) Remove common markdown code fences (```json ... ```)
    # recherche du premier bloc ```...``` contenant { ou [
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, flags=re.IGNORECASE)
    if fence_match:
        candidate = fence_match.group(1).strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            # continue to other attempts
            pass

    # 2) Cherche le premier objet JSON démarre par '[' ou '{' et parse tout jusqu'à sa fermeture correspondante
    # On trouve l'index du premier '[' ou '{'
    first_idx = None
    for ch in ('[', '{'):
        idx = text.find(ch)
        if idx != -1:
            if first_idx is None or idx < first_idx:
                first_idx = idx
    if first_idx is not None:
        candidate = text[first_idx:].strip()
        # Tentative progressive de découpage (par ex si le modèle ajoute du texte après le JSON)
        # On essaye de trouver la fin du JSON en comptant les crochets accolés
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            # Tentative: tronquer au dernier ']' ou '}' et retenter
            last_brace_idx = max(candidate.rfind(']'), candidate.rfind('}'))
            if last_brace_idx != -1:
                truncated = candidate[:last_brace_idx+1]
                try:
                    return json.loads(truncated)
                except json.JSONDecodeError:
                    pass

    # 3) fallback: rechercher des occurrences de petits JSONs via regex (pas parfait)
    simple_jsons = re.findall(r'(\{(?:[^{}]|(?R))*\}|\[(?:[^\[\]]|(?R))*\])', text)
    for s in simple_jsons:
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            continue

    # Si tout échoue, raise JSONDecodeError pour être géré par l'appelant
    raise json.JSONDecodeError("Impossible d'extraire un JSON valide", text, 0)

# --- LLM call ---------------------------------------------------------------
# Modèles OpenRouter recommandés pour ce type de tâche :
# meta-llama/llama-4-maverick:free
# deepseek/deepseek-r1-0528-qwen3-8b:free
# google/gemini-2.0-flash-exp:free
# mistralai/mistral-nemo:free

def call_llm_api(prompt, api_key, model_name="deepseek/deepseek-r1-0528-qwen3-8b:free"):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    # On envoie un message système pour contraindre la sortie à du JSON pur
    messages = [
        {"role": "system", "content": "Vous êtes un assistant qui répond STRICTEMENT par un JSON valide (liste d'objets). Ne fournissez aucun texte supplémentaire, aucune explication, aucune balise markdown."},
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

# --- Core processing -------------------------------------------------------
def process_profiles_with_llm(profiles, api_key, batch_size=50, output_dir="llm_outputs_deepseek"):
    os.makedirs(output_dir, exist_ok=True)
    all_matches = []
    # shuffle pour mélanger les sources (évite d'avoir des batches mono-source)
    random.seed(42)
    random.shuffle(profiles)

    batch_count = math.ceil(len(profiles) / batch_size)
    for idx, raw_batch in enumerate(chunk_list(profiles, batch_size)):
        batch_number = idx + 1
        print(f"[{datetime.utcnow().isoformat()}] Traitement batch {batch_number}/{batch_count} ({len(raw_batch)} profils)")

        # réduire les profils au minimum pour économiser tokens
        batch = [minimal_profile(p) for p in raw_batch]

        prompt = build_prompt(batch)
        try:
            result = call_llm_api(prompt, api_key)
            # chemin d'accès réponse (OpenRouter structure)
            llm_response_text = result['choices'][0]['message']['content']
        except Exception as e:
            # journaliser erreur API et continuer
            err_file = os.path.join(output_dir, f"batch_{batch_number}_api_error.txt")
            with open(err_file, "w", encoding="utf-8") as f:
                f.write(str(e))
            print(f"Erreur appel API pour batch {batch_number} : {e}")
            continue

        # sauvegarder la réponse brute
        raw_file = os.path.join(output_dir, f"batch_{batch_number}_raw.txt")
        with open(raw_file, "w", encoding="utf-8") as f:
            f.write(llm_response_text)

        # Essayer de parser le JSON même si response contient du texte annexe
        try:
            parsed = extract_json_from_text(llm_response_text)
            # si parsed est une liste de correspondances
            if isinstance(parsed, list):
                all_matches.extend(parsed)
            else:
                # on normalise en liste si le modèle a renvoyé un objet unique
                all_matches.append(parsed)
            # sauvegarder matches batch
            out_file = os.path.join(output_dir, f"batch_{batch_number}_matches.json")
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(parsed, f, indent=2, ensure_ascii=False)
            print(f"Batch {batch_number} parsé avec succès, {len(parsed) if isinstance(parsed, list) else 1} correspondances.")
        except json.JSONDecodeError:
            print(f"Erreur décodage JSON pour batch {batch_number}, réponse brute sauvegardée dans {raw_file}.")
            # Ici on pourrait tenter une stratégie de re-prompting ou envoyer un prompt de 'repair' au modèle.
            # Pour l'instant on continue.
            continue

    return all_matches

# --- Prompt builder --------------------------------------------------------
def build_prompt(profiles_batch):
    instruction = (
        "Vous recevez une liste de profils utilisateurs provenant de différentes plateformes "
        "(LinkedIn, Twitter, GitHub).\n"
        "Identifiez uniquement les correspondances entre profils provenant de plateformes différentes\n"
        "et retournez STRICTEMENT un JSON (liste) d'objets au format :\n"
        "[\n"
        "  {\"profile1\": {\"username\": \"...\", \"source\": \"LinkedIn\"}, \"profile2\": {\"username\": \"...\", \"source\": \"GitHub\"},\"profile3\": {\"username\": \"...\", \"source\": \"Twitter\"}, \"score\": 0.85},\n"
        "  ...\n"
        "]\n"
        "Ne renvoyez aucun texte, aucune explication, aucune balise markdown. Si vous ne trouvez aucune correspondance, répondez par [] (liste vide).\n"
    )
    # On inclut seulement les profils minimalistes (moins de tokens)
    profiles_json = json.dumps(profiles_batch, ensure_ascii=False, indent=2)
    prompt = f"{instruction}\nProfils :\n{profiles_json}\nRéponse :"
    return prompt

# --- File loader -----------------------------------------------------------
def load_json_with_source(filepath, source_name, limit=1000):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    data = data[:limit]
    for profile in data:
        profile['source'] = source_name
    return data

# --- Main ------------------------------------------------------------------
if __name__ == "__main__":
    API_KEY = os.getenv("openRouter_api_key")

    linkedin_profiles = load_json_with_source("linkedin_profiles_normalized.json", "LinkedIn", limit=1000)
    twitter_profiles = load_json_with_source("twitter_data_cleaned.json", "Twitter", limit=1000)
    github_profiles = load_json_with_source("github_cleaned.json", "GitHub", limit=1000)

    all_profiles = linkedin_profiles + twitter_profiles + github_profiles
    print(f"Nombre total de profils chargés : {len(all_profiles)}")

    matches = process_profiles_with_llm(all_profiles, API_KEY, batch_size=70, output_dir="llm_outputs_deepseek")

    with open("matched_profiles.json", "w", encoding="utf-8") as f:
        json.dump(matches, f, indent=2, ensure_ascii=False)

    print(f"Matching terminé, correspondances totales extraites : {len(matches)}")
