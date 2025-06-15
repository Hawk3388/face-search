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
import face_recognition
from PIL import Image
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from urllib.parse import urljoin, unquote

# ----------------------------------------
# 1. Utility: RAM-Monitoring
# ----------------------------------------
def memory_usage_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

# ----------------------------------------
# 2. Face-Encoding extrahieren
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

def get_query_encoding(input_image_path, resize_max=800):
    """
    Lädt das Eingabebild, extrahiert und gibt das Face-Encoding zurück.
    """
    if not os.path.exists(input_image_path):
        print(f"[get_query_encoding] Eingabebild nicht gefunden: {input_image_path}")
        return None
    try:
        img = Image.open(input_image_path).convert('RGB')
    except Exception as e:
        print(f"[get_query_encoding] Fehler beim Laden: {e}")
        return None
    enc = get_face_encoding_from_image(img, resize_max=resize_max)
    img.close()
    if enc is None:
        print("Kein Gesicht im Eingabebild erkannt.")
    return enc

# ----------------------------------------
# 3. Reverse Image Search bei Google (requests-basiert)
# ----------------------------------------
def google_reverse_image_search(image_path, user_agent=None, max_results=100):
    """
    Führt eine Reverse Image Search bei Google aus, indem das Bild per POST an den Upload-Endpoint gesendet wird.
    Parst anschließend die Ergebnisseite, um Bild-URLs zu extrahieren.
    Gibt eine Liste von URLs zurück (bis max_results), meist Thumbnails oder direkte Bild-URLs.
    """
    # 3.1. Session mit Headers anlegen
    session = requests.Session()
    # Setze User-Agent, falls nicht angegeben
    if user_agent is None:
        # Ein häufiger Desktop-User-Agent
        user_agent = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/90.0.4430.93 Safari/537.36")
    session.headers.update({
        "User-Agent": user_agent,
    })

    # 3.2. POST an Upload-Endpoint
    multipart = {
        'encoded_image': (os.path.basename(image_path), open(image_path, 'rb')),
        'image_content': ''
    }
    # Wichtig: allow_redirects=False, um den Redirect zu erhalten
    try:
        response = session.post(
            'https://www.google.com/searchbyimage/upload',
            files=multipart,
            allow_redirects=False,
            timeout=15
        )
    except Exception as e:
        print(f"[google_reverse_image_search] POST-Fehler: {e}")
        return []
    # 3.3. Google antwortet mit Redirect (302) zur Ergebnis-URL
    if response.status_code != 302 or 'Location' not in response.headers:
        print(f"[google_reverse_image_search] Unerwartete Antwort: Status {response.status_code}")
        return []
    fetch_url = response.headers['Location']
    # 3.4. GET die Ergebnisseite
    try:
        resp2 = session.get(fetch_url, timeout=15)
    except Exception as e:
        print(f"[google_reverse_image_search] GET-Ergebnisseite-Fehler: {e}")
        return []
    if resp2.status_code != 200:
        print(f"[google_reverse_image_search] Fehler beim Laden der Ergebnisseite: Status {resp2.status_code}")
        return []

    html = resp2.text
    # 3.5. Parse HTML mit BeautifulSoup
    soup = BeautifulSoup(html, 'html.parser')
    image_urls = set()

    # 3.6. Versuche, Bild-URLs aus der Seite zu extrahieren
    # Google zeigt Thumbnails in <img> mit class "rg_i" oder "t0fcAb" u.a.
    # Wir suchen nach <img> Tags mit data-src oder src, und ggf. JSON in Seite.
    # Dies ist best-effort und kann angepasst werden.

    # 3.6.1. Suche nach <img> Tags mit src/data-src
    for img_tag in soup.find_all('img'):
        # Manche Thumbnails haben 'data-src' oder 'src'
        src = img_tag.get('data-src') or img_tag.get('src')
        if not src:
            continue
        # Filtere unwahrscheinliche Mini-Icons
        if src.startswith('data:'):  # embedded small icons
            continue
        # Vollständige URL? Wenn relativ, baue vollständige URL
        if src.startswith('//'):
            src = 'https:' + src
        elif src.startswith('/'):
            src = urljoin('https://www.google.com', src)
        # Füge hinzu
        image_urls.add(src)
        if len(image_urls) >= max_results:
            break

    # 3.6.2. Manchmal sind in der Seite JSON-Blöcke mit "ou":"<image_url>"
    # Wir können nach solchen Mustern suchen:
    if len(image_urls) < max_results:
        text = html
        # Suche rudimentär nach ou":"URL" Mustern
        import re
        pattern = re.compile(r'"ou":"([^"]+)"')
        for match in pattern.finditer(text):
            url = match.group(1)
            # Ent-escape
            url = url.replace('\\u003d', '=').replace('\\u0026', '&')
            if url.startswith('http'):
                image_urls.add(url)
            if len(image_urls) >= max_results:
                break

    # 3.6.3. Weitere Ansätze: JSON-LD oder Skript-Blöcke parsen, falls erforderlich.

    return list(image_urls)[:max_results]

# ----------------------------------------
# 4. Sequenzielle Verarbeitung der gefundenen URLs
# ----------------------------------------
async def process_candidate_urls_sequential(urls, query_encoding, top_k=5, resize_max=800):
    """
    Lädt sequenziell URLs herunter, extrahiert Face-Encoding, vergleicht Distanz.
    Gibt Liste [(dist, url), ...] sortiert zurück (aufsteigende Distanz).
    """
    top_matches = []
    timeout = aiohttp.ClientTimeout(total=10)
    connector = aiohttp.TCPConnector(limit_per_host=5)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
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

            enc = get_face_encoding_from_image(img, resize_max=resize_max)
            img.close()
            if enc is None:
                continue

            dist = np.linalg.norm(enc - query_encoding)
            # Top-K pflegen
            if len(top_matches) < top_k:
                top_matches.append((dist, url))
                top_matches.sort(key=lambda x: x[0])
            else:
                if dist < top_matches[-1][0]:
                    top_matches[-1] = (dist, url)
                    top_matches.sort(key=lambda x: x[0])

            # Monitoring
            if (i + 1) % 20 == 0:
                print(f"  Verarbeitet {i+1}/{len(urls)} URLs, bester Distanz so weit: {top_matches[0][0]:.3f}, RAM: {memory_usage_mb():.1f} MB")
            # Optional: Stop-Kriterium, falls dist < Threshold: break

    return top_matches

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
if __name__ == "__main__":
    # 6.1. Parameter anpassen
    input_path = "input.jpg"   # Dein Query-Gesicht
    max_results = 100          # Wie viele Bild-URLs wir aus der Google-Ergebnisseite extrahieren
    top_k = 5                  # Anzahl finaler Treffer
    resize_max = 800           # Resize vor Face-Erkennung

    # 6.2. Query-Encoding extrahieren
    query_enc = get_query_encoding(input_path, resize_max=resize_max)
    if query_enc is None:
        exit(1)

    # 6.3. Reverse Image Search bei Google
    print("Starte Reverse Image Search bei Google...")
    urls = google_reverse_image_search(input_path, max_results=max_results)
    print(f"Extrahierte URLs: {len(urls)}")
    # Anzeige extrahierter URLs (optional):
    # for u in urls[:10]:
    #     print(u)

    if not urls:
        print("Keine URLs extrahiert, beende.")
        exit(1)

    # 6.4. Sequenzielle Verarbeitung
    print("Verarbeite URLs sequenziell für Face-Matching...")
    top_matches = asyncio.run(process_candidate_urls_sequential(urls, query_enc, top_k=top_k, resize_max=resize_max))

    # 6.5. Ausgabe der Top-K
    print("Top Matches (aufsteigende Distanz):")
    for idx, (dist, url) in enumerate(top_matches, 1):
        print(f"  {idx}: {url} mit Distanz {dist:.3f}")

    # 6.6. Thumbnails anzeigen
    show_top_matches(top_matches)

    print("Fertig.")
