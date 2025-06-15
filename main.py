#!/usr/bin/env python3
"""
Ressourcenschonende Face-Matching-Pipeline.

Funktionen:
  - Input-Face-Encoding extrahieren (aus lokalem Bild).
  - Kandidaten lokal oder per URL sequenziell verarbeiten, auf Gesichter prüfen, Embeddings extrahieren.
  - Top-K ähnlichste Bilder (lokal oder URLs) anhand Embedding-Distanz finden und anzeigen.
  - Monitoring der RAM-Nutzung.
  - Bild-Resize vor der Erkennung, um RAM zu sparen.

Voraussetzungen (lokal installieren):
  pip install face_recognition aiohttp requests pillow numpy matplotlib psutil
  (Optional: wenn du nicht alle Module hast, installiere jene, die du brauchst. face_recognition erfordert dlib etc.)

Nutzung:
  - Kopiere dieses Skript in eine Datei, z.B. face_pipeline.py.
  - Passe ggf. Pfade, Parameter (resize_max, top_k) an.
  - Führe lokal aus.
"""

import os
import io
import asyncio
import requests
import aiohttp
import psutil
import numpy as np
import face_recognition
from PIL import Image
import matplotlib.pyplot as plt

# ----------------------------------------
# 1. Utility: RAM-Monitoring
# ----------------------------------------
def memory_usage_mb():
    """Gibt aktuellen RSS-RAM-Verbrauch des Prozesses in MB zurück."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

# ----------------------------------------
# 2. Funktion: Face-Encoding aus Bild extrahieren
# ----------------------------------------
def get_face_encoding_from_image(image: Image.Image, resize_max=800):
    """
    Erkennt das erste Gesicht in einem PIL-Image und gibt das 128D-Embedding zurück.
    - image: PIL.Image, RGB.
    - resize_max: maximale Größe der längeren Seite; Bilder werden verkleinert, falls größer.
    Rückgabe:
      - numpy array (128D) bei erkanntem Gesicht
      - None, wenn kein Gesicht erkannt wurde.
    """
    # Resize, um RAM und Rechenzeit zu sparen
    w, h = image.size
    if max(w, h) > resize_max:
        scale = resize_max / max(w, h)
        new_size = (int(w * scale), int(h * scale))
        image = image.resize(new_size, Image.LANCZOS)

    # In NumPy-Array konvertieren
    img_array = np.array(image)

    # Face-Lokation und Embedding
    locations = face_recognition.face_locations(img_array)
    if not locations:
        return None
    encodings = face_recognition.face_encodings(img_array, locations)
    if not encodings:
        return None
    return encodings[0]

# ----------------------------------------
# 3. Lokale Bilder sequenziell verarbeiten
# ----------------------------------------
def find_similar_local_images(input_image_path, candidate_folder, top_k=5, resize_max=800):
    """
    Verarbeitet Kandidatenbilder in einem lokalen Ordner sequenziell:
    - Extrahiere das Face-Embedding aus input_image_path.
    - Für jede Bilddatei im Ordner:
      - Bild laden, auf resize_max verkleinern, Gesicht enkodieren.
      - Distanz zum Query-Embedding berechnen.
      - Nur ein Gesicht pro Bild wird angenommen (erste erkannte).
    - Sortiere nach Distanz, zeige Top-K Pfade und Thumbnails mit Distanz.
    - Gibt eine Liste [(distance, path), ...] für Top-K zurück.
    """
    # 1. Input-Face laden und enkodieren
    try:
        input_image = Image.open(input_image_path).convert('RGB')
    except Exception as e:
        print(f"[find_similar_local_images] Fehler beim Laden des Input-Bilds: {e}")
        return []
    query_encoding = get_face_encoding_from_image(input_image, resize_max=resize_max)
    input_image.close()
    if query_encoding is None:
        print("Kein Gesicht im Eingabebild erkannt.")
        return []

    # 2. Kandidatenverzeichnis durchlaufen
    records = []  # Liste von (dist, path)
    file_list = sorted(os.listdir(candidate_folder))
    count = 0
    for fname in file_list:
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        candidate_path = os.path.join(candidate_folder, fname)
        try:
            img = Image.open(candidate_path).convert('RGB')
        except Exception as e:
            print(f"[find_similar_local_images] Fehler beim Laden von {candidate_path}: {e}")
            continue

        enc = get_face_encoding_from_image(img, resize_max=resize_max)
        img.close()  # Bildobjekt schließen, um RAM freizugeben
        if enc is None:
            continue

        # Distanz berechnen
        dist = np.linalg.norm(enc - query_encoding)
        records.append((dist, candidate_path))
        count += 1
        # Optionales Monitoring
        if count % 50 == 0:
            print(f"  Verarbeitete Gesichter: {count}, aktueller RAM: {memory_usage_mb():.1f} MB")

    if not records:
        print("Keine Gesichter in den Kandidatenbildern gefunden.")
        return []

    # 3. Sortieren und Top-K auswählen
    records.sort(key=lambda x: x[0])
    top_records = records[:top_k]

    # 4. Ausgabe Pfade + Distanz
    print(f"Top {top_k} Treffer:")
    for idx, (dist, path) in enumerate(top_records, 1):
        print(f"  {idx}: {path} mit Distanz {dist:.3f}")

    # 5. Thumbnails anzeigen
    cols = min(top_k, 5)
    rows = (top_k + cols - 1) // cols
    plt.figure(figsize=(3 * cols, 3 * rows))
    for i, (dist, path) in enumerate(top_records):
        try:
            img = Image.open(path).convert('RGB')
        except:
            continue
        img.thumbnail((200, 200), Image.LANCZOS)
        ax = plt.subplot(rows, cols, i + 1)
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(f"{dist:.2f}")
        img.close()
    plt.tight_layout()
    plt.show()

    return top_records

# ----------------------------------------
# 4. URL-basierte sequenzielle Verarbeitung
# ----------------------------------------
async def process_candidate_urls_sequential(urls, query_encoding, top_k=5, resize_max=800):
    """
    Verarbeitet eine Liste von Bild-URLs sequenziell:
    - Download pro URL, prüfe Content-Type (muss Bild sein).
    - Lade Bild in PIL, verkleinere auf resize_max, Gesicht enkodieren.
    - Berechne Distanz zum query_encoding.
    - Halte Top-K Treffer in einer Liste.
    - Gibt Liste [(dist, url), ...] sortiert zurück.
    Dieses Vorgehen lädt nie alle Bilder gleichzeitig in den RAM.
    """
    top_matches = []  # Liste von (dist, url), sortiert nach dist aufsteigend
    timeout = aiohttp.ClientTimeout(total=10)
    connector = aiohttp.TCPConnector(limit_per_host=10)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        for i, url in enumerate(urls):
            try:
                async with session.get(url) as resp:
                    if resp.status != 200:
                        continue
                    content_type = resp.headers.get('Content-Type', '')
                    if 'image' not in content_type:
                        continue
                    data = await resp.read()
                    img = Image.open(io.BytesIO(data)).convert('RGB')
            except Exception:
                continue

            enc = get_face_encoding_from_image(img, resize_max=resize_max)
            img.close()
            if enc is None:
                continue

            dist = np.linalg.norm(enc - query_encoding)
            # In Top-K einpflegen
            if len(top_matches) < top_k:
                top_matches.append((dist, url))
                top_matches.sort(key=lambda x: x[0])
            else:
                # Wenn besser als aktuell schlechtester, ersetzen
                if dist < top_matches[-1][0]:
                    top_matches[-1] = (dist, url)
                    top_matches.sort(key=lambda x: x[0])

            # Monitoring & evtl. frühes Abbrechen
            if (i + 1) % 50 == 0:
                print(f"  Verarbeitet {i+1} URLs, bester Distanz so weit: {top_matches[0][0]:.3f}, RAM: {memory_usage_mb():.1f} MB")
            # Optional: hier könnte man ein Stop-Kriterium setzen, z.B. wenn Distanz < Schwelle erreicht.

    return top_matches

# ----------------------------------------
# 5. Hilfsfunktion: Query-Encoding extrahieren
# ----------------------------------------
def get_query_encoding(input_image_path, resize_max=800):
    """
    Extrahiere und gib das Embedding des ersten Gesichts im Eingabebild zurück.
    """
    try:
        img = Image.open(input_image_path).convert('RGB')
    except Exception as e:
        print(f"[get_query_encoding] Fehler beim Laden des Input-Bilds: {e}")
        return None
    enc = get_face_encoding_from_image(img, resize_max=resize_max)
    img.close()
    if enc is None:
        print("Kein Gesicht im Eingabebild erkannt.")
    return enc

# ----------------------------------------
# 6. Beispiel: Verwendung lokal
# ----------------------------------------
if __name__ == "__main__":
    # Beispiel für lokale Bilder:
    # Lege input.jpg und Ordner 'candidates/' mit Bildern an.
    input_path = "input.jpg"
    candidates_dir = "candidates"
    top_k = 5
    resize_max = 800

    # Extrahiere Query-Encoding
    query_enc = get_query_encoding(input_path, resize_max=resize_max)
    if query_enc is not None:
        # Lokale Verarbeitung
        print("Starte lokale Suche...")
        top_local = find_similar_local_images(input_path, candidates_dir, top_k=top_k, resize_max=resize_max)
        # top_local enthält Liste von (dist, path)
        # Weitere Verarbeitung möglich...

    # Beispiel für URL-Listen (pseudocode; erfordert eigene URL-Liste)
    # urls = ["https://.../img1.jpg", "https://.../img2.jpg", ...]
    # if query_enc is not None and urls:
    #     print("Starte URL-basierte Suche...")
    #     results = asyncio.run(process_candidate_urls_sequential(urls, query_enc, top_k=top_k, resize_max=resize_max))
    #     print("Top-K Treffer aus URLs:")
    #     for idx, (dist, url) in enumerate(results, 1):
    #         print(f"  {idx}: {url} mit Distanz {dist:.3f}")

    print("Pipeline beendet. Passe Pfade/Parameter für eigene Nutzung.")

# ----------------------------------------
# Ende des Skripts
# ----------------------------------------
