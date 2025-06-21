#!/usr/bin/env python3
"""
Automated Reverse Image Search using Google without API key and without Selenium.
Followed by sequential face recognition and matching.
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
# 2. Face Recognition and VGG16-based Feature Extraction
# ----------------------------------------

def get_face_encoding_from_image(image: Image.Image, resize_max=800):
    """
    Detects the first face in a PIL image and returns the 128D embedding, or None.
    Resizes the image to resize_max (largest side) to save RAM/CPU.
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
    Loads the input image, extracts and returns the face encoding.
    """
    if not os.path.exists(input_image_path):
        print(f"[get_query_face_encoding] Input image not found: {input_image_path}")
        return None
    try:
        img = Image.open(input_image_path).convert('RGB')
    except Exception as e:
        print(f"[get_query_face_encoding] Error loading: {e}")
        return None
    enc = get_face_encoding_from_image(img, resize_max=resize_max)
    img.close()
    if enc is None:
        print("No face detected in input image.")
    return enc

# Global VGG16 model
_vgg16_model = None

def get_vgg16_model():
    """Loads the VGG16 model once"""
    global _vgg16_model
    if _vgg16_model is None:
        print("Loading VGG16 model...")
        _vgg16_model = VGG16(weights="imagenet", include_top=False, pooling="avg")
        print("VGG16 model loaded!")
    return _vgg16_model

def extract_vgg16_features(image: Image.Image, target_size=(224, 224)):
    """
    Extracts VGG16 features from a PIL image
    """
    try:
        # Resize image to VGG16 input size
        img = image.resize(target_size, Image.LANCZOS)
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        
        # Extract features
        model = get_vgg16_model()
        features = model.predict(img_array, verbose=0)
        return features.flatten()
    except Exception as e:
        print(f"Error in feature extraction: {e}")
        return None

def get_query_features(input_image_path):
    """
    Loads the input image and extracts VGG16 features
    """
    if not os.path.exists(input_image_path):
        print(f"[get_query_features] Input image not found: {input_image_path}")
        return None
    try:
        img = Image.open(input_image_path).convert('RGB')
        features = extract_vgg16_features(img)
        img.close()
        if features is None:
            print("Error in feature extraction from input image.")
        return features
    except Exception as e:
        print(f"[get_query_features] Error loading: {e}")
        return None

# ----------------------------------------
# 3. PicImageSearch-based Reverse Image Search
# ----------------------------------------

def extract_urls_from_google_response(resp: GoogleResponse) -> list:
    """Extracts all image URLs from Google response"""
    urls = set()
    
    # Check available attributes
    if hasattr(resp, 'results') and resp.results:
        for item in resp.results:
            if hasattr(item, 'url') and item.url:
                urls.add(item.url)
            if hasattr(item, 'thumbnail') and item.thumbnail:
                urls.add(item.thumbnail)
    
    # Alternative attribute names
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
    """Extracts all image URLs from Bing response"""
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
    Performs Google Reverse Image Search with PicImageSearch
    """
    print(f"[GOOGLE-PIC] === Starting Google Reverse Search ===")
    print(f"[GOOGLE-PIC] Image Path: {image_path}")
    print(f"[GOOGLE-PIC] Max Results: {max_results}")
    
    try:
        async with Network(proxies=None) as client:
            google = Google(client=client)
            resp = await google.search(file=image_path)
            
            print(f"[GOOGLE-PIC] Search URL: {resp.url}")
            
            urls = extract_urls_from_google_response(resp)
            print(f"[GOOGLE-PIC] Found URLs: {len(urls)}")
            
            return urls[:max_results]
            
    except Exception as e:
        print(f"[GOOGLE-PIC] ✗ Error: {e}")
        logger.exception("Error in google_reverse_image_search:")
        return []

async def bing_reverse_image_search(image_path, max_results=100):
    """
    Performs Bing Reverse Image Search with PicImageSearch
    """
    print(f"[BING-PIC] === Starting Bing Reverse Search ===")
    print(f"[BING-PIC] Image Path: {image_path}")
    print(f"[BING-PIC] Max Results: {max_results}")
    
    try:
        async with Network(proxies=None) as client:
            bing = Bing(client=client)
            resp = await bing.search(file=image_path)
            
            print(f"[BING-PIC] Search URL: {resp.url}")
            
            urls = extract_urls_from_bing_response(resp)
            print(f"[BING-PIC] Found URLs: {len(urls)}")
            
            return urls[:max_results]
            
    except Exception as e:
        print(f"[BING-PIC] ✗ Error: {e}")
        logger.exception("Error in bing_reverse_image_search:")
        return []

def google_reverse_image_search_sync(image_path, max_results=100):
    """
    Synchronous Google Reverse Image Search with PicImageSearch
    """
    print(f"[GOOGLE-SYNC] === Starting Google Reverse Search (Sync) ===")
    print(f"[GOOGLE-SYNC] Image Path: {image_path}")
    print(f"[GOOGLE-SYNC] Max Results: {max_results}")
    
    try:
        google = GoogleSync(proxies=None)
        resp = google.search(file=image_path)
        
        print(f"[GOOGLE-SYNC] Search URL: {resp.url}")
        
        urls = extract_urls_from_google_response(resp)
        print(f"[GOOGLE-SYNC] Found URLs: {len(urls)}")
        
        return urls[:max_results]
        
    except Exception as e:
        print(f"[GOOGLE-SYNC] ✗ Error: {e}")
        logger.exception("Error in google_reverse_image_search_sync:")
        return []

def bing_reverse_image_search_sync(image_path, max_results=100):
    """
    Synchronous Bing Reverse Image Search with PicImageSearch
    """
    print(f"[BING-SYNC] === Starting Bing Reverse Search (Sync) ===")
    print(f"[BING-SYNC] Image Path: {image_path}")
    print(f"[BING-SYNC] Max Results: {max_results}")
    
    try:
        bing = BingSync(proxies=None)
        resp = bing.search(file=image_path)
        
        print(f"[BING-SYNC] Search URL: {resp.url}")
        
        urls = extract_urls_from_bing_response(resp)
        print(f"[BING-SYNC] Found URLs: {len(urls)}")
        
        return urls[:max_results]
        
    except Exception as e:
        print(f"[BING-SYNC] ✗ Error: {e}")
        logger.exception("Error in bing_reverse_image_search_sync:")
        return []

async def bing_only_reverse_search(image_path, max_results=200):
    """
    Performs only Bing Reverse Search (Google removed)
    """
    print(f"[BING-ONLY] === Starting Bing Reverse Search ===")
    print(f"[BING-ONLY] Using only Bing with PicImageSearch")
    
    # Bing Search
    print("\n1. Starting Bing Search...")
    bing_start = time.time()
    bing_urls = await bing_reverse_image_search(image_path, max_results)
    bing_time = time.time() - bing_start
    print(f"[BING-ONLY] Bing: {len(bing_urls)} URLs in {bing_time:.2f}s")
    
    print(f"\n[BING-ONLY] === Bing Search Completed ===")
    print(f"[BING-ONLY] Total URLs: {len(bing_urls)}")
    
    return bing_urls

# ----------------------------------------
# 4. Two-stage processing: VGG16 + Face-Recognition
# ----------------------------------------
async def process_candidate_urls_two_stage(urls, query_features, query_face_encoding, vgg16_threshold=0.85, top_k=20):
    """
    Two-stage approach:
    1. VGG16-Filter: REMOVE images with >85% general similarity (too similar/identical)
    2. Face-Recognition: Sort ALL filtered images by face similarity
    
    Returns list [(face_similarity, url), ...] sorted (descending face similarity).
    """
    stage1_candidates = []
    stage2_matches = []
    timeout = aiohttp.ClientTimeout(total=10)
    connector = aiohttp.TCPConnector(limit_per_host=5)
    
    print(f"=== TWO-STAGE FILTER ===")
    print(f"Stage 1: VGG16-Filter (REMOVES images with >{vgg16_threshold*100:.0f}% image similarity)")
    print(f"Stage 2: Face-Recognition (ALL remaining faces sorted by similarity)")
    print(f"Processing {len(urls)} URLs...")
    
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        # STAGE 1: VGG16-Filter (REVERSE - removes too similar images)
        print(f"\n--- STAGE 1: VGG16-Filter (removes too similar images) ---")
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

            # Extract VGG16 features
            features = extract_vgg16_features(img)
            if features is None:
                img.close()
                continue

            # Calculate VGG16 similarity
            try:
                vgg16_similarity = cosine_similarity([query_features], [features])[0][0]
            except Exception:
                img.close()
                continue
            
            # Stage 1 Filter: Only images UNDER VGG16 threshold (less similar)
            if vgg16_similarity < vgg16_threshold:
                stage1_candidates.append((vgg16_similarity, url, img))
                if len(stage1_candidates) % 10 == 0:
                    print(f"  Stage 1: {len(stage1_candidates)} candidates kept (from {i+1} processed URLs)")
            else:
                img.close()
                # These images are filtered out (too similar)

            # Monitoring
            if (i + 1) % 50 == 0:
                print(f"  Processed {i+1}/{len(urls)} URLs, kept: {len(stage1_candidates)} (removed too similar)")
                print(f"✅ Stage 1 completed: {len(stage1_candidates)} candidates under {vgg16_threshold*100:.0f}% VGG16 similarity kept")
        
        if not stage1_candidates:
            print("❌ All images were too similar to original (above VGG16 threshold)!")
            return []

        # ADDITIONAL STEP: VGG16 duplicate filtering between candidates
        print(f"\n--- ADDITIONAL: VGG16 duplicate filtering between candidates ---")
        filtered_candidates = []
        candidate_features = []
        
        # Extract features of all candidates
        for i, (vgg16_sim, url, img) in enumerate(stage1_candidates):
            features = extract_vgg16_features(img)
            if features is not None:
                candidate_features.append(features)
                filtered_candidates.append((vgg16_sim, url, img, features))
        
        # Remove similar images among each other
        unique_candidates = []
        used_indices = set()
        
        for i, (vgg16_sim, url, img, features) in enumerate(filtered_candidates):
            if i in used_indices:
                continue
                
            is_unique = True
            for j, (_, _, _, other_features) in enumerate(filtered_candidates[:i]):
                if j in used_indices:
                    continue
                    
                try:
                    similarity = cosine_similarity([features], [other_features])[0][0]
                    if similarity >= vgg16_threshold:  # Too similar to an already selected image
                        is_unique = False
                        break
                except Exception:
                    continue
            
            if is_unique:
                unique_candidates.append((vgg16_sim, url, img))
            else:
                img.close()  # Close similar image
                used_indices.add(i)
        
        print(f"✅ Duplicate filtering: {len(stage1_candidates)} → {len(unique_candidates)} unique candidates")

        # STAGE 2: Face-Recognition on ALL filtered candidates
        print(f"\n--- STAGE 2: Face-Recognition (sort all faces) ---")
        for i, (vgg16_sim, url, img) in enumerate(unique_candidates):
            # Extract face encoding
            face_encoding = get_face_encoding_from_image(img)
            img.close()
            
            if face_encoding is None:
                continue  # No face detected

            # Calculate face similarity (distance -> similarity)
            try:
                face_distance = np.linalg.norm(face_encoding - query_face_encoding)
                face_similarity = 1 / (1 + face_distance)  # Convert distance to similarity
            except Exception:
                continue
            
            # Collect ALL faces (no threshold)
            stage2_matches.append((face_similarity, url))            # Monitoring
            if (i + 1) % 10 == 0:
                print(f"  Stage 2: {i+1}/{len(unique_candidates)} processed, faces found: {len(stage2_matches)}")

        # Sort by face similarity (descending order)
        stage2_matches.sort(key=lambda x: x[0], reverse=True)
        
        # Remove duplicates (same URLs)
        seen_urls = set()
        unique_matches = []
        for face_similarity, url in stage2_matches:
            if url not in seen_urls:
                seen_urls.add(url)
                unique_matches.append((face_similarity, url))
        
        if len(stage2_matches) != len(unique_matches):
            print(f"  🔄 {len(stage2_matches) - len(unique_matches)} duplicates removed")
        
        # Limit to top-K
        if len(unique_matches) > top_k:
            unique_matches = unique_matches[:top_k]

        print(f"✅ Stage 2 completed: {len(unique_matches)} unique faces found and sorted")
        if unique_matches:
            best_face_sim = unique_matches[0][0] * 100
            worst_face_sim = unique_matches[-1][0] * 100
            print(f"  Best face similarity: {best_face_sim:.1f}%")
            print(f"  Worst face similarity: {worst_face_sim:.1f}%")
        
        return unique_matches

# ----------------------------------------
# 5. Display of top matches
# ----------------------------------------
def show_top_matches(top_matches):
    """
    Downloads the top match URLs again and shows thumbnails with URLs.
    """
    if not top_matches:
        print("No matches to display.")
        return
    
    print(f"\n🖼️  Loading {len(top_matches)} images for thumbnail display...")
    cols = min(len(top_matches), 5)
    rows = (len(top_matches) + cols - 1)//cols
    plt.figure(figsize=(4*cols, 4*rows))
    
    successful_images = 0
    
    for i, (similarity, url) in enumerate(top_matches):
        print(f"  Loading image {i+1}/{len(top_matches)}: {url[:60]}...")
        try:
            resp = requests.get(url, timeout=10, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            if resp.status_code != 200:
                print(f"    ❌ HTTP {resp.status_code}")
                continue
            
            img = Image.open(io.BytesIO(resp.content)).convert('RGB')
            print(f"    ✅ Successfully loaded ({img.size})")
            
        except requests.exceptions.Timeout:
            print(f"    ❌ Timeout while loading")
            continue
        except requests.exceptions.RequestException as e:
            print(f"    ❌ Network error: {e}")
            continue
        except Exception as e:
            print(f"    ❌ Image error: {e}")
            continue
            
        img.thumbnail((200, 200), Image.LANCZOS)
        ax = plt.subplot(rows, cols, i+1)
        ax.imshow(img)
        ax.axis('off')
        
        # Show similarity and URL
        similarity_percent = similarity * 100
        # Shorten URL for better display (domain + path)
        from urllib.parse import urlparse
        parsed_url = urlparse(url)
        short_url = f"{parsed_url.netloc}"
        if parsed_url.path:
            path_parts = parsed_url.path.split('/')
            if len(path_parts) > 1:
                short_url += f"/.../{path_parts[-1]}"
        
        # Display text below the image
        ax.text(0.5, -0.15, f"{similarity_percent:.1f}%\n{short_url}", 
                transform=ax.transAxes, ha='center', va='top',
                fontsize=8, wrap=True)
        
        successful_images += 1
    
    print(f"\n✅ {successful_images}/{len(top_matches)} images successfully loaded")
    
    if successful_images > 0:
        plt.tight_layout()
        plt.show()
    else:
        print("❌ No images could be loaded!")
        print("Possible causes:")
        print("- Images are no longer available")
        print("- Servers block access")
        print("- Network problems")

# ----------------------------------------
# 6. Main Program
# ----------------------------------------
if __name__ == "__main__":
    # 6.1. Adjust parameters
    # Input image path with validation
    while True:
        input_path = input("Path to image: ")
        input_path = input_path.replace('"', '').replace(" ", "").strip()

        if os.path.exists(input_path):
            # Check if it's an image file
            if input_path.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff')):
                print(f"✅ Image found: {input_path}")
                break
            else:
                print(f"❌ The file '{input_path}' is not a supported image file!")
                print("Supported formats: .jpg, .jpeg, .png, .gif, .bmp, .webp, .tiff")
                continue
        else:
            print(f"❌ Path '{input_path}' not found!")
            print(f"Current directory: {os.getcwd()}")
            print("Available image files:")
            found_images = False
            for f in os.listdir('.'):
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff')):
                    print(f"  - {f}")
                    found_images = True
            if not found_images:
                print("  (No image files found in current directory)")
            print("Please enter a valid path.\n")    
    max_results = 200          # Increased for better results
    top_k = 10                 # More final matches
    vgg16_threshold = 0.85     # 85% VGG16 similarity for prefilter
    face_threshold = 0.0       # No face threshold - sort all faces
    
    print(f"=== Face Search Tool with PicImageSearch ===")
    print(f"Using input image: {input_path}")
    print(f"File size: {os.path.getsize(input_path)} bytes")

    # 6.2. Extract query features and face encoding
    print("\n=== STEP 1: Analyze query image ===")
    
    # VGG16 features for prefilter
    query_features = get_query_features(input_path)
    if query_features is None:
        print("❌ Error extracting VGG16 features from input image!")
        exit(1)
    print("✅ VGG16 features successfully extracted!")
    
    # Face encoding for final face comparisons
    query_face_encoding = get_query_face_encoding(input_path)
    if query_face_encoding is None:
        print("❌ No face detected in input image!")
        exit(1)
    print("✅ Face successfully detected!")

    # 6.3. Bing Reverse Image Search (Bing only)
    print("\n=== STEP 2: Bing Reverse Image Search ===")
    print("Using PicImageSearch with Bing only...")
    search_start_time = time.time()
    urls = asyncio.run(bing_only_reverse_search(input_path, max_results=max_results))
    search_total_time = time.time() - search_start_time
    
    print(f"\n[MAIN] Bing search completed in {search_total_time:.2f}s")
    print(f"[MAIN] Extracted URLs: {len(urls)}")
    
    # Analyze search results
    if urls:
        print(f"\n=== URL ANALYSIS ===")
        print(f"First 10 URLs:")
        for i, u in enumerate(urls[:10], 1):
            print(f"  {i:2d}: {u}")
        if len(urls) > 10:
            print(f"  ... and {len(urls)-10} more URLs")
        
        # Domain analysis
        from urllib.parse import urlparse
        url_domains = {}
        for url in urls:
            try:
                domain = urlparse(url).netloc
                url_domains[domain] = url_domains.get(domain, 0) + 1
            except:
                pass
        
        print(f"\nURL sources:")
        for domain, count in sorted(url_domains.items(), key=lambda x: x[1], reverse=True)[:8]:
            print(f"  {domain}: {count} URLs")
    
    if not urls:
        print("\n❌ NO URLs found via Bing Reverse Image Search!")
        print("Possible causes:")
        print("1. The image is very unusual or not available on the internet")
        print("2. The person is not well-known enough for image databases")
        print("3. Bing is blocking our requests")
        print("4. The input image has poor quality")
        print("\nRECOMMENDATIONS:")
        print("- Use a different image of the person")
        print("- Try an image with better resolution")
        print("- Use an image of a more well-known person for testing")
        exit(1)
    
    # 6.4. Two-stage processing: VGG16 + Face-Recognition
    print("\n=== STEP 3: Two-stage matching ===")
    print("Using VGG16 prefilter + Face-Recognition...")
    matching_start_time = time.time()
    top_matches = asyncio.run(process_candidate_urls_two_stage(
        urls, 
        query_features, 
        query_face_encoding,
        vgg16_threshold=vgg16_threshold,
        top_k=top_k
    ))
    
    matching_total_time = time.time() - matching_start_time
    print(f"\n[MAIN] Two-stage matching completed in {matching_total_time:.2f}s")

    # 6.5. Output of top-K results
    print("\n=== STEP 4: Results ===")
    print(f"=== Top Matches (descending face similarity) ===")
    
    if not top_matches:
        print("❌ NO MATCHES FOUND ABOVE THRESHOLDS!")
        print("Possible causes:")
        print(f"1. No images reach the VGG16 threshold of {vgg16_threshold*100:.0f}% similarity")
        print(f"2. No faces reach the face threshold of {face_threshold*100:.0f}% similarity")
        print("3. Found URLs contain no valid images or faces")
        print("4. The query image is too specific or unusual")
        print("\nSOLUTIONS:")
        print("- Reduce the thresholds (e.g. VGG16: 60%, Face: 75%)")
        print("- Use a different image")
        print("- Increase max_results for more candidates")
    else:
        print(f"✅ {len(top_matches)} matches found and sorted by face similarity!")
        print("\n" + "="*80)
        print("RESULTS (descending face similarity):")
        print("="*80)
        
        for idx, (face_similarity, url) in enumerate(top_matches, 1):
            face_similarity_percent = face_similarity * 100
            print(f"\n🏆 MATCH #{idx}:")
            print(f"   Face similarity: {face_similarity_percent:.1f}%")
            print(f"   Image URL: {url}")
            print(f"   {'-'*60}")
        
        print(f"\n✅ Best face match: {top_matches[0][0]*100:.1f}% similarity")
        
        if top_matches[0][0] < 0.90:  # Under 90%
            print("⚠️ NOTE: Even with high similarity, these could be different people.")
        else:
            print("✅ Very high face similarity values found!")

    # 6.6. Display thumbnails
    print("\n=== STEP 5: Thumbnail Display ===")
    show_top_matches(top_matches)
    
    print("\n✅ Analysis completed!")
