#!/usr/bin/env python3
"""
Automatisierte Reverse Image Search bei Google ohne API-Key und ohne Selenium.
Anschließend sequenzielle Gesichtserkennung und Matching.

WARNUNG:
- Das Scrapen von Google kann gegen die Nutzungsbedingungen verstoßen und zu temporären Blockierungen führen.
- Verwende dieses Skript vorsichtig und in Maßen (Pausen, respektiere Robots.txt, setze angemessene User-Agent-Header).
- Google ändert seine HTML-Struktur regelmäßig, daher kann der Parsing-Teil abbrechen und muss ggf. angepasst werden.
"""

import os
import io
import time
import requests
import asyncio
import aiohttp
import psutil
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

# TensorFlow imports
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

# Face Recognition imports
import face_recognition

from loguru import logger

# PicImageSearch imports
from PicImageSearch import Google, Bing, Network
from PicImageSearch.model import GoogleResponse, BingResponse
from PicImageSearch.sync import Google as GoogleSync, Bing as BingSync

# ----------------------------------------
# 1. Utility: RAM-Monitoring
# ----------------------------------------
def memory_usage_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

# ----------------------------------------
# 2. Gesichtserkennung und VGG16-basierte Feature-Extraktion
# ----------------------------------------

def get_face_encoding_from_image(image: Image.Image, resize_max=800):
    """
    Erkennt das erste Gesicht in einem PIL-Image und gibt das 128D-Embedding zurück, oder None.
    Verkleinert das Bild auf resize_max (größte Seite), um RAM/CPU zu sparen.
    """
    w, h = image.size
    if max(w, h) > resize_max:
        scale = resize_max / max(w, h)
        image = image.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
    img_array = np.array(image)
    locations = face_recognition.face_locations(img_array)
    if not locations:
        return None
    encodings = face_recognition.face_encodings(img_array, locations)
    if not encodings:
        return None
    return encodings[0]

def get_query_face_encoding(input_image_path, resize_max=800):
    """
    Lädt das Eingabebild, extrahiert und gibt das Face-Encoding zurück.
    """
    if not os.path.exists(input_image_path):
        print(f"[get_query_face_encoding] Eingabebild nicht gefunden: {input_image_path}")
        return None
    try:
        img = Image.open(input_image_path).convert('RGB')
    except Exception as e:
        print(f"[get_query_face_encoding] Fehler beim Laden: {e}")
        return None
    enc = get_face_encoding_from_image(img, resize_max=resize_max)
    img.close()
    if enc is None:
        print("Kein Gesicht im Eingabebild erkannt.")
    return enc

# Globales VGG16-Modell
_vgg16_model = None

def get_vgg16_model():
    """Lädt das VGG16-Modell einmalig"""
    global _vgg16_model
    if _vgg16_model is None:
        print("Lade VGG16-Modell...")
        _vgg16_model = VGG16(weights="imagenet", include_top=False, pooling="avg")
        print("VGG16-Modell geladen!")
    return _vgg16_model

def extract_vgg16_features(image: Image.Image, target_size=(224, 224)):
    """
    Extrahiert VGG16-Features aus einem PIL-Image
    """
    try:
        # Bild auf VGG16-Eingabegröße anpassen
        img = image.resize(target_size, Image.LANCZOS)
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        
        # Features extrahieren
        model = get_vgg16_model()
        features = model.predict(img_array, verbose=0)
        return features.flatten()
    except Exception as e:
        print(f"Fehler bei Feature-Extraktion: {e}")
        return None

def get_query_features(input_image_path):
    """
    Lädt das Eingabebild und extrahiert VGG16-Features
    """
    if not os.path.exists(input_image_path):
        print(f"[get_query_features] Eingabebild nicht gefunden: {input_image_path}")
        return None
    try:
        img = Image.open(input_image_path).convert('RGB')
        features = extract_vgg16_features(img)
        img.close()
        if features is None:
            print("Fehler bei der Feature-Extraktion vom Eingabebild.")
        return features
    except Exception as e:
        print(f"[get_query_features] Fehler beim Laden: {e}")
        return None

# ----------------------------------------
# 3. PicImageSearch-basierte Reverse Image Search
# ----------------------------------------

def extract_urls_from_google_response(resp: GoogleResponse) -> list:
    """Extrahiert alle Bild-URLs aus der Google-Response"""
    urls = set()
    
    # Prüfe verfügbare Attribute
    if hasattr(resp, 'results') and resp.results:
        for item in resp.results:
            if hasattr(item, 'url') and item.url:
                urls.add(item.url)
            if hasattr(item, 'thumbnail') and item.thumbnail:
                urls.add(item.thumbnail)
    
    # Alternative Attribute-Namen  
    for attr_name in ['same', 'similar', 'related', 'visually_similar']:
        if hasattr(resp, attr_name):
            items = getattr(resp, attr_name)
            if items:
                for item in items:
                    if hasattr(item, 'url') and item.url:
                        urls.add(item.url)
                    if hasattr(item, 'thumbnail') and item.thumbnail:
                        urls.add(item.thumbnail)
                
    return list(urls)

def extract_urls_from_bing_response(resp: BingResponse) -> list:
    """Extrahiert alle Bild-URLs aus der Bing-Response"""
    urls = set()
    
    # Pages including
    if resp.pages_including:
        for item in resp.pages_including:
            if hasattr(item, 'image_url') and item.image_url:
                urls.add(item.image_url)
            if hasattr(item, 'thumbnail') and item.thumbnail:
                urls.add(item.thumbnail)
    
    # Visual search
    if resp.visual_search:
        for item in resp.visual_search:
            if hasattr(item, 'image_url') and item.image_url:
                urls.add(item.image_url)
            if hasattr(item, 'thumbnail') and item.thumbnail:
                urls.add(item.thumbnail)
                
    return list(urls)

async def google_reverse_image_search(image_path, max_results=100):
    """
    Führt Google Reverse Image Search mit PicImageSearch aus
    """
    print(f"[GOOGLE-PIC] === Starte Google Reverse Search ===")
    print(f"[GOOGLE-PIC] Bild-Pfad: {image_path}")
    print(f"[GOOGLE-PIC] Max Results: {max_results}")
    
    try:
        async with Network(proxies=None) as client:
            google = Google(client=client)
            resp = await google.search(file=image_path)
            
            print(f"[GOOGLE-PIC] Search URL: {resp.url}")
            
            urls = extract_urls_from_google_response(resp)
            print(f"[GOOGLE-PIC] Gefundene URLs: {len(urls)}")
            
            return urls[:max_results]
            
    except Exception as e:
        print(f"[GOOGLE-PIC] ✗ Fehler: {e}")
        logger.exception("Fehler in google_reverse_image_search:")
        return []

async def bing_reverse_image_search(image_path, max_results=100):
    """
    Führt Bing Reverse Image Search mit PicImageSearch aus
    """
    print(f"[BING-PIC] === Starte Bing Reverse Search ===")
    print(f"[BING-PIC] Bild-Pfad: {image_path}")
    print(f"[BING-PIC] Max Results: {max_results}")
    
    try:
        async with Network(proxies=None) as client:
            bing = Bing(client=client)
            resp = await bing.search(file=image_path)
            
            print(f"[BING-PIC] Search URL: {resp.url}")
            
            urls = extract_urls_from_bing_response(resp)
            print(f"[BING-PIC] Gefundene URLs: {len(urls)}")
            
            return urls[:max_results]
            
    except Exception as e:
        print(f"[BING-PIC] ✗ Fehler: {e}")
        logger.exception("Fehler in bing_reverse_image_search:")
        return []

def google_reverse_image_search_sync(image_path, max_results=100):
    """
    Synchrone Google Reverse Image Search mit PicImageSearch
    """
    print(f"[GOOGLE-SYNC] === Starte Google Reverse Search (Sync) ===")
    print(f"[GOOGLE-SYNC] Bild-Pfad: {image_path}")
    print(f"[GOOGLE-SYNC] Max Results: {max_results}")
    
    try:
        google = GoogleSync(proxies=None)
        resp = google.search(file=image_path)
        
        print(f"[GOOGLE-SYNC] Search URL: {resp.url}")
        
        urls = extract_urls_from_google_response(resp)
        print(f"[GOOGLE-SYNC] Gefundene URLs: {len(urls)}")
        
        return urls[:max_results]
        
    except Exception as e:
        print(f"[GOOGLE-SYNC] ✗ Fehler: {e}")
        logger.exception("Fehler in google_reverse_image_search_sync:")
        return []

def bing_reverse_image_search_sync(image_path, max_results=100):
    """
    Synchrone Bing Reverse Image Search mit PicImageSearch
    """
    print(f"[BING-SYNC] === Starte Bing Reverse Search (Sync) ===")
    print(f"[BING-SYNC] Bild-Pfad: {image_path}")
    print(f"[BING-SYNC] Max Results: {max_results}")
    
    try:
        bing = BingSync(proxies=None)
        resp = bing.search(file=image_path)
        
        print(f"[BING-SYNC] Search URL: {resp.url}")
        
        urls = extract_urls_from_bing_response(resp)
        print(f"[BING-SYNC] Gefundene URLs: {len(urls)}")
        
        return urls[:max_results]
        
    except Exception as e:
        print(f"[BING-SYNC] ✗ Fehler: {e}")
        logger.exception("Fehler in bing_reverse_image_search_sync:")
        return []

async def bing_only_reverse_search(image_path, max_results=200):
    """
    Führt nur Bing Reverse Search aus (Google entfernt)
    """
    print(f"[BING-ONLY] === Starte Bing Reverse Search ===")
    print(f"[BING-ONLY] Verwende nur Bing mit PicImageSearch")
    
    # Bing Search
    print("\n1. Starte Bing Search...")
    bing_start = time.time()
    bing_urls = await bing_reverse_image_search(image_path, max_results)
    bing_time = time.time() - bing_start
    print(f"[BING-ONLY] Bing: {len(bing_urls)} URLs in {bing_time:.2f}s")
    
    print(f"\n[BING-ONLY] === Bing Search Abgeschlossen ===")
    print(f"[BING-ONLY] Gesamt URLs: {len(bing_urls)}")
    
    return bing_urls

# ----------------------------------------
# 4. Zweistufige Verarbeitung: VGG16 + Face-Recognition
# ----------------------------------------
async def process_candidate_urls_two_stage(urls, query_features, query_face_encoding, vgg16_threshold=0.85, top_k=20):
    """
    Zweistufiger Ansatz:
    1. VGG16-Filter: Bilder mit >85% allgemeiner Ähnlichkeit RAUSFILTERN (zu ähnlich/identisch)
    2. Face-Recognition: ALLE gefilterten Bilder nach Gesichtsähnlichkeit sortieren
    
    Gibt Liste [(face_similarity, url), ...] sortiert zurück (absteigende Gesichtsähnlichkeit).
    """
    stage1_candidates = []
    stage2_matches = []
    timeout = aiohttp.ClientTimeout(total=10)
    connector = aiohttp.TCPConnector(limit_per_host=5)
    
    print(f"=== ZWEISTUFIGER FILTER ===")
    print(f"Stufe 1: VGG16-Filter (ENTFERNT Bilder mit >{vgg16_threshold*100:.0f}% Bildähnlichkeit)")
    print(f"Stufe 2: Face-Recognition (ALLE verbleibenden Gesichter nach Ähnlichkeit sortieren)")
    print(f"Verarbeite {len(urls)} URLs...")
    
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        # STUFE 1: VGG16-Filter (UMGEKEHRT - entfernt zu ähnliche Bilder)
        print(f"\n--- STUFE 1: VGG16-Filter (entfernt zu ähnliche Bilder) ---")
        for i, url in enumerate(urls):
            try:
                async with session.get(url) as resp:
                    if resp.status != 200:
                        continue
                    ctype = resp.headers.get("Content-Type", "")
                    if "image" not in ctype:
                        continue
                    data = await resp.read()
                    img = Image.open(io.BytesIO(data)).convert('RGB')
            except Exception:
                continue

            # VGG16-Features extrahieren
            features = extract_vgg16_features(img)
            if features is None:
                img.close()
                continue

            # VGG16-Ähnlichkeit berechnen
            try:
                vgg16_similarity = cosine_similarity([query_features], [features])[0][0]
            except Exception:
                img.close()
                continue
            
            # Stufe 1 Filter: Nur Bilder UNTER dem VGG16-Schwellenwert (weniger ähnlich)
            if vgg16_similarity < vgg16_threshold:
                stage1_candidates.append((vgg16_similarity, url, img))
                if len(stage1_candidates) % 10 == 0:
                    print(f"  Stufe 1: {len(stage1_candidates)} Kandidaten behalten (von {i+1} verarbeiteten URLs)")
            else:
                img.close()
                # Diese Bilder werden rausgefiltert (zu ähnlich)

            # Monitoring
            if (i + 1) % 50 == 0:
                print(f"  Verarbeitet {i+1}/{len(urls)} URLs, behalten: {len(stage1_candidates)} (entfernt zu ähnliche)")

        print(f"✅ Stufe 1 abgeschlossen: {len(stage1_candidates)} Kandidaten unter {vgg16_threshold*100:.0f}% VGG16-Ähnlichkeit behalten")
        
        if not stage1_candidates:
            print("❌ Alle Bilder waren zu ähnlich zum Original (über dem VGG16-Schwellenwert)!")
            return []

        # STUFE 2: Face-Recognition auf ALLE gefilterten Kandidaten
        print(f"\n--- STUFE 2: Face-Recognition (alle Gesichter sortieren) ---")
        for i, (vgg16_sim, url, img) in enumerate(stage1_candidates):
            # Face-Encoding extrahieren
            face_encoding = get_face_encoding_from_image(img)
            img.close()
            
            if face_encoding is None:
                continue  # Kein Gesicht erkannt

            # Face-Ähnlichkeit berechnen (Distanz -> Ähnlichkeit)
            try:
                face_distance = np.linalg.norm(face_encoding - query_face_encoding)
                face_similarity = 1 / (1 + face_distance)  # Distanz zu Ähnlichkeit konvertieren
            except Exception:
                continue
            
            # ALLE Gesichter sammeln (kein Schwellenwert)
            stage2_matches.append((face_similarity, url))

            # Monitoring
            if (i + 1) % 10 == 0:
                print(f"  Stufe 2: {i+1}/{len(stage1_candidates)} verarbeitet, Gesichter gefunden: {len(stage2_matches)}")

        # Nach Gesichtsähnlichkeit sortieren (absteigende Reihenfolge)
        stage2_matches.sort(key=lambda x: x[0], reverse=True)
        
        # Top-K begrenzen
        if len(stage2_matches) > top_k:
            stage2_matches = stage2_matches[:top_k]

        print(f"✅ Stufe 2 abgeschlossen: {len(stage2_matches)} Gesichter gefunden und sortiert")
        if stage2_matches:
            best_face_sim = stage2_matches[0][0] * 100
            worst_face_sim = stage2_matches[-1][0] * 100
            print(f"  Beste Gesichtsähnlichkeit: {best_face_sim:.1f}%")
            print(f"  Schlechteste Gesichtsähnlichkeit: {worst_face_sim:.1f}%")
        
        return stage2_matches

# ----------------------------------------
# 5. Anzeige der Top-Treffer
# ----------------------------------------
def show_top_matches(top_matches):
    """
    Lädt die Top-Treffer-URLs erneut herunter und zeigt Thumbnails mit Distanz.
    """
    if not top_matches:
        print("Keine Treffer zum Anzeigen.")
        return
    cols = min(len(top_matches), 5)
    rows = (len(top_matches) + cols - 1)//cols
    plt.figure(figsize=(3*cols, 3*rows))
    for i, (dist, url) in enumerate(top_matches):
        try:
            resp = requests.get(url, timeout=5)
            if resp.status_code != 200:
                continue
            img = Image.open(io.BytesIO(resp.content)).convert('RGB')
        except Exception:
            continue
        img.thumbnail((200, 200), Image.LANCZOS)
        ax = plt.subplot(rows, cols, i+1)
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(f"{dist:.2f}")
    plt.tight_layout()
    plt.show()

# ----------------------------------------
# 6. Hauptprogramm
# ----------------------------------------
if __name__ == "__main__":    # 6.1. Parameter anpassen
    input_path = "./images/input.jpg"   # Dein Query-Bild
    max_results = 200          # Erhöht für bessere Ergebnisse
    top_k = 10                 # Mehr finale Treffer
    vgg16_threshold = 0.75     # 85% VGG16-Ähnlichkeit für Vorfilter
    face_threshold = 0.0       # Kein Face-Schwellenwert - alle Gesichter sortieren

    # 6.1.5. Überprüfe Eingabedatei
    if not os.path.exists(input_path):
        print(f"Fehler: Eingabedatei '{input_path}' nicht gefunden!")
        print(f"Aktuelles Verzeichnis: {os.getcwd()}")
        print("Verfügbare Dateien:")
        for f in os.listdir('.'):
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
                print(f"  - {f}")
        exit(1)
    
    print(f"=== Face Search Tool mit PicImageSearch ===")
    print(f"Verwende Eingabebild: {input_path}")
    print(f"Dateigröße: {os.path.getsize(input_path)} Bytes")    # 6.2. Query-Features und Face-Encoding extrahieren
    print("\n=== SCHRITT 1: Query-Bild analysieren ===")
    
    # VGG16-Features für Vorfilter
    query_features = get_query_features(input_path)
    if query_features is None:
        print("❌ Fehler bei der VGG16-Feature-Extraktion vom Eingabebild!")
        exit(1)
    print("✅ VGG16-Features erfolgreich extrahiert!")
    
    # Face-Encoding für finale Gesichtsvergleiche
    query_face_encoding = get_query_face_encoding(input_path)
    if query_face_encoding is None:
        print("❌ Kein Gesicht im Eingabebild erkannt!")
        exit(1)
    print("✅ Gesicht erfolgreich erkannt!")    # 6.3. Bing Reverse Image Search (nur Bing)
    print("\n=== SCHRITT 2: Bing Reverse Image Search ===")
    print("Verwende PicImageSearch nur mit Bing...")
    
    search_start_time = time.time()
    urls = asyncio.run(bing_only_reverse_search(input_path, max_results=max_results))
    search_total_time = time.time() - search_start_time
    
    print(f"\n[HAUPTPROGRAMM] Bing Search abgeschlossen in {search_total_time:.2f}s")
    print(f"[HAUPTPROGRAMM] Extrahierte URLs: {len(urls)}")
    
    # Analysiere die Suchergebnisse
    if urls:
        print(f"\n=== URL-ANALYSE ===")
        print(f"Erste 10 URLs:")
        for i, u in enumerate(urls[:10], 1):
            print(f"  {i:2d}: {u}")
        if len(urls) > 10:
            print(f"  ... und {len(urls)-10} weitere URLs")
        
        # Domain-Analyse
        from urllib.parse import urlparse
        url_domains = {}
        for url in urls:
            try:
                domain = urlparse(url).netloc
                url_domains[domain] = url_domains.get(domain, 0) + 1
            except:
                pass
        
        print(f"\nURL-Quellen:")
        for domain, count in sorted(url_domains.items(), key=lambda x: x[1], reverse=True)[:8]:
            print(f"  {domain}: {count} URLs")
    
    if not urls:
        print("\n❌ KEINE URLs über Bing Reverse Image Search gefunden!")
        print("Mögliche Ursachen:")
        print("1. Das Bild ist sehr ungewöhnlich oder nicht im Internet verfügbar")
        print("2. Die Person ist nicht bekannt genug für Bilddatenbanken")
        print("3. Bing blockiert unsere Anfragen")
        print("4. Das Eingabebild hat schlechte Qualität")
        print("\nEMPFEHLUNGEN:")
        print("- Verwenden Sie ein anderes Bild der Person")
        print("- Probieren Sie ein Bild mit besserer Auflösung")
        print("- Verwenden Sie ein Bild von einer bekannteren Person zum Testen")
        exit(1)
    
    # 6.4. Zweistufige Verarbeitung: VGG16 + Face-Recognition
    print("\n=== SCHRITT 3: Zweistufiges Matching ===")
    print("Verwende VGG16-Vorfilter + Face-Recognition...")
    matching_start_time = time.time()
    
    top_matches = asyncio.run(process_candidate_urls_two_stage(
        urls, 
        query_features, 
        query_face_encoding,
        vgg16_threshold=vgg16_threshold,
        top_k=top_k
    ))
    
    matching_total_time = time.time() - matching_start_time
    print(f"\n[HAUPTPROGRAMM] Zweistufiges Matching abgeschlossen in {matching_total_time:.2f}s")

    # 6.5. Ausgabe der Top-K
    print("\n=== SCHRITT 4: Ergebnisse ===")
    print(f"=== Top Matches (absteigende Gesichtsähnlichkeit) ===")
    
    if not top_matches:
        print("❌ KEINE MATCHES ÜBER DEN SCHWELLENWERTEN GEFUNDEN!")
        print("Mögliche Ursachen:")
        print(f"1. Keine Bilder erreichen den VGG16-Schwellenwert von {vgg16_threshold*100:.0f}% Ähnlichkeit")
        print(f"2. Keine Gesichter erreichen den Face-Schwellenwert von {face_threshold*100:.0f}% Ähnlichkeit")
        print("3. Die gefundenen URLs enthalten keine gültigen Bilder oder Gesichter")
        print("4. Das Query-Bild ist zu spezifisch oder ungewöhnlich")
        print("\nLÖSUNGSANSÄTZE:")
        print("- Reduzieren Sie die Schwellenwerte (z.B. VGG16: 60%, Face: 75%)")
        print("- Verwenden Sie ein anderes Bild")
        print("- Erhöhen Sie max_results für mehr Kandidaten")
    else:
        print(f"✅ {len(top_matches)} Matches über {face_threshold*100:.0f}% Gesichtsähnlichkeit gefunden!")
        for idx, (face_similarity, url) in enumerate(top_matches, 1):
            face_similarity_percent = face_similarity * 100
            print(f"  {idx}: Gesichtsähnlichkeit {face_similarity_percent:.1f}%")
            print(f"      URL: {url}")
        
        print(f"\nBeste Gesichtsübereinstimmung: {top_matches[0][0]*100:.1f}% Ähnlichkeit")
        
        if top_matches[0][0] < 0.90:  # Unter 90%
            print("⚠️ HINWEIS: Auch bei hoher Ähnlichkeit könnten es verschiedene Personen sein.")
        else:
            print("✅ Sehr hohe Gesichtsähnlichkeitswerte gefunden!")

    # 6.6. Thumbnails anzeigen
    print("\n=== SCHRITT 5: Thumbnail-Anzeige ===")
    show_top_matches(top_matches)
    
    print("\n✅ Analyse abgeschlossen!")
