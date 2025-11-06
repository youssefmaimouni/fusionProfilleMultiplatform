import json
import math
# Ajout de la bibliothèque Google
import google.generativeai as genai
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Découpage d'une liste de profils en batchs de taille batch_size
def chunk_list(data, batch_size):
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]

# Construction du prompt à envoyer au LLM avec un batch de profils
# Le prompt reste inchangé, il est bien formulé.
def build_prompt(profiles_batch):
    instruction = (
        "Vous recevez une liste de profils utilisateurs provenant de différentes plateformes "
        "(LinkedIn, Twitter, GitHub). "
        "Identifiez uniquement les correspondances entre profils provenant de plateformes différentes "
        "et retournez un JSON listant les paires de profils correspondants avec un score.\n"
    )
    # Pour Gemini, il est préférable de ne pas inclure "Réponse :" à la fin du prompt.
    # Le modèle est entraîné pour répondre directement à l'instruction.
    profiles_json = json.dumps(profiles_batch, ensure_ascii=False)
    prompt = f"{instruction}\nProfils :\n{profiles_json}"
    return prompt

# ---- FONCTION MODIFIÉE POUR GOOGLE AI STUDIO (GEMINI) ----
# Fonction d'appel à l'API LLM (Google AI Studio)
def call_llm_api(prompt, api_key, model_name="gemini-pro-latest"):
    """
    Appelle l'API Gemini de Google AI Studio.
    """
    try:
        # Configuration de la clé API
        genai.configure(api_key=api_key)

        # Création du modèle
        model = genai.GenerativeModel(model_name)

        # Envoi du prompt au modèle
        response = model.generate_content(prompt)
        
        # Le contenu généré se trouve dans response.text
        return response.text
        
    except Exception as e:
        # Gérer les erreurs potentielles de l'API
        print(f"Erreur lors de l'appel à l'API Gemini : {e}")
        return None

# Fonction principale pour traiter tous les profils batch par batch
def process_profiles_with_llm(profiles, api_key, batch_size=100):
    all_matches = []
    batch_count = math.ceil(len(profiles) / batch_size)
    for idx, batch in enumerate(chunk_list(profiles, batch_size)):
        print(f"Traitement batch {idx + 1}/{batch_count} avec {len(batch)} profils")
        prompt = build_prompt(batch)
        llm_response_text = call_llm_api(prompt, api_key)

        if llm_response_text:
            # Gemini peut retourner le JSON dans un bloc de code Markdown.
            # On nettoie la réponse pour extraire le JSON pur.
            if llm_response_text.strip().startswith("```json"):
                llm_response_text = llm_response_text.strip()[7:-3].strip()

            try:
                matches = json.loads(llm_response_text)
                all_matches.extend(matches)
            except json.JSONDecodeError:
                print(f"Erreur décodage JSON pour batch {idx + 1}, réponse brute : {llm_response_text}")
    return all_matches

# Chargement des profils avec ajout de la source
def load_json_with_source(filepath, source_name):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    for profile in data:
        profile['source'] = source_name
    return data

# Exemple d'usage complet
if __name__ == "__main__":
    # Remplacez par votre clé API de Google AI Studio
    # Il est recommandé de la stocker dans une variable d'environnement pour plus de sécurité
    API_KEY = os.getenv("gemini_api_key")

    linkedin_profiles = load_json_with_source("linkedin_profiles_normalized.json", "LinkedIn")
    twitter_profiles = load_json_with_source("twitter_data_cleaned.json", "Twitter")
    github_profiles = load_json_with_source("github_cleaned.json", "GitHub")

    # Fusionner toutes les données dans une liste
    all_profiles = linkedin_profiles + twitter_profiles + github_profiles

    matches = process_profiles_with_llm(all_profiles, API_KEY, batch_size=100)

    with open("matched_profiles_gemini.json", "w", encoding="utf-8") as f:
        json.dump(matches, f, indent=2, ensure_ascii=False)

    print(f"Matching terminé, nombre total de correspondances extraites : {len(matches)}")