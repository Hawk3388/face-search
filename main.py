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
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

# Suppress TensorFlow logging unless in debug mode
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow INFO and WARNING messages
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# TensorFlow imports
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

# Face Recognition imports
import face_recognition

from loguru import logger

# PicImageSearch imports
from PicImageSearch import Bing, Tineye, Network
from PicImageSearch.model import BingResponse, TineyeResponse

# ----------------------------------------
# 1. Utility Functions
# ----------------------------------------

def show_progress_bar(current, total, step_name="Processing", width=50):
    """Shows a progress bar for the current step"""
    if total == 0:
        return
    
    percent = current / total
    filled = int(width * percent)
    bar = '█' * filled + '░' * (width - filled)
    
    # Clear line and show progress
    print(f'\r{step_name}: [{bar}] {current}/{total} ({percent:.1%})', end='', flush=True)
    
    if current == total:
        print()  # New line when complete

def animate_step(step_name, duration=1.0):
    """Shows an animated loading indicator for a step"""
    chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    end_time = time.time() + duration
    i = 0
    
    while time.time() < end_time:
        print(f'\r{chars[i % len(chars)]} {step_name}...', end='', flush=True)
        time.sleep(0.1)
        i += 1
    
    print(f'\r✅ {step_name} completed!           ')

# ----------------------------------------
# 2. Face Recognition and VGG16-based Feature Extraction
# ----------------------------------------

def get_face_encoding_from_image(image_path, resize_max=800):
    """
    Detects the first face in an image file and returns the 128D embedding, or None.
    """
    try:
        # Load image using face_recognition
        img = face_recognition.load_image_file(image_path)
        
        # Get face locations and encodings
        locations = face_recognition.face_locations(img, model='cnn')
        if not locations:
            return None
        
        encodings = face_recognition.face_encodings(img, locations)
        if not encodings:
            return None
        
        return encodings[0]
    except Exception:
        return None

def get_face_encoding_from_pil_image(pil_image):
    """
    Detects the first face in a PIL image and returns the 128D embedding, or None.
    """
    try:
        # Convert PIL image to numpy array for face_recognition
        img_array = np.array(pil_image)
        
        # Get face locations and encodings
        locations = face_recognition.face_locations(img_array)
        if not locations:
            return None
        
        encodings = face_recognition.face_encodings(img_array, locations)
        if not encodings:
            return None
        
        return encodings[0]
    except Exception:
        return None

# Global VGG16 model
_vgg16_model = None

def get_vgg16_model(debug=False):
    """Loads the VGG16 model once"""
    global _vgg16_model
    if _vgg16_model is None:
        if debug:
            print("Loading VGG16 model...")
        _vgg16_model = VGG16(weights="imagenet", include_top=False, pooling="avg")
        if debug:
            print("VGG16 model loaded!")
    return _vgg16_model

def extract_vgg16_features(image: Image.Image, target_size=(224, 224), debug=False):
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
        model = get_vgg16_model(debug)
        features = model.predict(img_array, verbose=0)
        return features.flatten()
    except Exception as e:
        if debug:
            print(f"Error in feature extraction: {e}")
        return None

def get_query_features(input_image_path, debug=False):
    """
    Loads the input image and extracts VGG16 features
    """
    if not os.path.exists(input_image_path):
        if debug:
            print(f"[get_query_features] Input image not found: {input_image_path}")
        return None
    try:
        img = Image.open(input_image_path).convert('RGB')
        features = extract_vgg16_features(img, debug=debug)
        img.close()
        if features is None and debug:
            print("Error in feature extraction from input image.")
        return features
    except Exception as e:
        if debug:
            print(f"[get_query_features] Error loading: {e}")
        return None

# ----------------------------------------
# 3. PicImageSearch-based Reverse Image Search
# ----------------------------------------

def extract_urls_from_bing_response(resp: BingResponse) -> list:
    """Extracts image URLs and their source page URLs from Bing response"""
    url_pairs = []  # List of (image_url, source_page_url) tuples
    
    # Pages including
    if resp.pages_including:
        for item in resp.pages_including:
            image_url = None
            source_url = None
            
            # Get image URL (prefer original over thumbnail)
            if hasattr(item, 'image_url') and item.image_url:
                image_url = item.image_url
            elif hasattr(item, 'thumbnail') and item.thumbnail:
                image_url = item.thumbnail
            
            # Get source page URL
            if hasattr(item, 'url') and item.url:
                source_url = item.url
            elif hasattr(item, 'source') and item.source:
                source_url = item.source
            
            if image_url and source_url:
                url_pairs.append((image_url, source_url))
    
    # Visual search
    if resp.visual_search:
        for item in resp.visual_search:
            image_url = None
            source_url = None
            
            # Get image URL (prefer original over thumbnail)
            if hasattr(item, 'image_url') and item.image_url:
                image_url = item.image_url
            elif hasattr(item, 'thumbnail') and item.thumbnail:
                image_url = item.thumbnail
            
            # Get source page URL
            if hasattr(item, 'url') and item.url:
                source_url = item.url
            elif hasattr(item, 'source') and item.source:
                source_url = item.source
            
            if image_url and source_url:
                url_pairs.append((image_url, source_url))
                
    return url_pairs

async def bing_reverse_image_search(image_path, max_results=100, debug=False):
    """
    Performs Bing Reverse Image Search with PicImageSearch
    """
    if debug:
        print(f"[BING-PIC] === Starting Bing Reverse Search ===")
        print(f"[BING-PIC] Image Path: {image_path}")
        print(f"[BING-PIC] Max Results: {max_results}")
    
    try:
        async with Network(proxies=None) as client:
            bing = Bing(client=client)
            resp = await bing.search(file=image_path)
            
            if debug:
                print(f"[BING-PIC] Search URL: {resp.url}")
            
            url_pairs = extract_urls_from_bing_response(resp)
            if debug:
                print(f"[BING-PIC] Found URL pairs: {len(url_pairs)}")
            
            return url_pairs[:max_results]
            
    except Exception as e:
        print(f"[BING-PIC] ✗ Error: {e}")
        logger.exception("Error in bing_reverse_image_search:")
        return []

async def bing_with_tineye_fallback_search(image_path, max_results=200, debug=False):
    """
    Performs Bing Reverse Search first, then TinEye as fallback if no results
    """
    if debug:
        print(f"[SEARCH] === Starting Reverse Image Search ===")
        print(f"[SEARCH] Strategy: Bing first, TinEye fallback")
    
    # Try Bing Search first
    if debug:
        print("\n1. Starting Bing Search...")
    bing_start = time.time()
    bing_url_pairs = await bing_reverse_image_search(image_path, max_results, debug)
    bing_time = time.time() - bing_start
    if debug:
        print(f"[SEARCH] Bing: {len(bing_url_pairs)} URL pairs in {bing_time:.2f}s")
    
    # If Bing found results, use them
    if bing_url_pairs:
        if debug:
            print(f"[SEARCH] ✅ Bing found {len(bing_url_pairs)} results, using Bing results")
            print(f"[SEARCH] === Search Completed (Bing) ===")
        return bing_url_pairs
    
    # If Bing found nothing, try TinEye as fallback
    if debug:
        print(f"[SEARCH] ⚠️ Bing found no results, trying TinEye as fallback...")
        print("\n2. Starting TinEye Search...")
    
    tineye_start = time.time()
    tineye_url_pairs = await tineye_reverse_image_search(image_path, max_results, debug)
    tineye_time = time.time() - tineye_start
    if debug:
        print(f"[SEARCH] TinEye: {len(tineye_url_pairs)} URL pairs in {tineye_time:.2f}s")
    
    if tineye_url_pairs:
        if debug:
            print(f"[SEARCH] ✅ TinEye found {len(tineye_url_pairs)} results, using TinEye results")
            print(f"[SEARCH] === Search Completed (TinEye Fallback) ===")
        return tineye_url_pairs
    else:
        if debug:
            print(f"[SEARCH] ❌ Both Bing and TinEye found no results")
            print(f"[SEARCH] === Search Completed (No Results) ===")
        return []

def extract_urls_from_tineye_response(resp: TineyeResponse) -> list:
    """Extracts image URLs and their source page URLs from TinEye response"""
    url_pairs = []  # List of (image_url, source_page_url) tuples
    
    if resp and resp.raw:
        for item in resp.raw:
            image_url = None
            source_url = None
            
            # Get image URL
            if hasattr(item, 'image_url') and item.image_url:
                image_url = item.image_url
            elif hasattr(item, 'thumbnail') and item.thumbnail:
                image_url = item.thumbnail
            
            # Get source page URL (backlink)
            if hasattr(item, 'url') and item.url:
                source_url = item.url
                
            if image_url and source_url:
                url_pairs.append((image_url, source_url))
                
    return url_pairs

async def tineye_reverse_image_search(image_path, max_results=100, debug=False):
    """
    Performs TinEye Reverse Image Search with PicImageSearch
    """
    if debug:
        print(f"[TINEYE] === Starting TinEye Reverse Search ===")
        print(f"[TINEYE] Image Path: {image_path}")
        print(f"[TINEYE] Max Results: {max_results}")
    
    try:
        async with Network(proxies=None) as client:
            tineye = Tineye(client=client)
            resp = await tineye.search(
                file=image_path,
                show_unavailable_domains=False,
                domain="",
                tags="",
                sort="score",
                order="desc",
            )
            
            if debug:
                if resp and hasattr(resp, 'query_hash'):
                    print(f"[TINEYE] Query Hash: {resp.query_hash}")
                if resp and hasattr(resp, 'total_pages'):
                    print(f"[TINEYE] Total Pages: {resp.total_pages}")
            
            url_pairs = extract_urls_from_tineye_response(resp)
            if debug:
                print(f"[TINEYE] Found URL pairs: {len(url_pairs)}")
            
            return url_pairs[:max_results]
            
    except Exception as e:
        print(f"[TINEYE] ✗ Error: {e}")
        logger.exception("Error in tineye_reverse_image_search:")
        return []

# ----------------------------------------
# 4. Two-stage processing: VGG16 + Face-Recognition
# ----------------------------------------
async def process_candidate_urls_two_stage(url_pairs, query_features, query_face_encoding, vgg16_threshold=0.85, top_k=20, debug=False):
    """
    Two-stage approach:
    1. VGG16-Filter: REMOVE images with >85% general similarity (too similar/identical)
    2. Face-Recognition: Sort ALL filtered images by face similarity
    
    Returns list [(face_similarity, source_page_url), ...] sorted (descending face similarity).
    """
    stage1_candidates = []
    stage2_matches = []
    timeout = aiohttp.ClientTimeout(total=10)
    connector = aiohttp.TCPConnector(limit_per_host=5)
    
    if debug:
        print(f"=== TWO-STAGE FILTER ===")
        print(f"Stage 1: VGG16-Filter (REMOVES images with >{vgg16_threshold*100:.0f}% image similarity)")
        print(f"Stage 2: Face-Recognition (ALL remaining faces sorted by similarity)")
        print(f"Processing {len(url_pairs)} URL pairs...")
    
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:        # STAGE 1: VGG16-Filter (REVERSE - removes too similar images)
        if debug:
            print(f"\n--- STAGE 1: VGG16-Filter (removes too similar images) ---")
        
        total_pairs = len(url_pairs)
        for i, (image_url, source_page_url) in enumerate(url_pairs):
            if not debug:
                show_progress_bar(i + 1, total_pairs, "Stage 1: VGG16 filtering")
                
            try:
                async with session.get(image_url) as resp:
                    if resp.status != 200:
                        continue
                    ctype = resp.headers.get("Content-Type", "")
                    if "image" not in ctype:
                        continue
                    data = await resp.read()
                    img = Image.open(io.BytesIO(data)).convert('RGB')
            except Exception:
                continue            # Extract VGG16 features
            features = extract_vgg16_features(img, debug=debug)
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
                stage1_candidates.append((vgg16_similarity, image_url, source_page_url, img))
                if debug and len(stage1_candidates) % 10 == 0:
                    print(f"  Stage 1: {len(stage1_candidates)} candidates kept (from {i+1} processed URL pairs)")
            else:
                img.close()
                # These images are filtered out (too similar)            # Monitoring
            if debug and (i + 1) % 50 == 0:
                print(f"  Processed {i+1}/{len(url_pairs)} URL pairs, kept: {len(stage1_candidates)} (removed too similar)")
        
        if debug:
            print(f"✅ Stage 1 completed: {len(stage1_candidates)} candidates under {vgg16_threshold*100:.0f}% VGG16 similarity kept")
        
        if not stage1_candidates:
            if debug:
                print("❌ All images were too similar to original (above VGG16 threshold)!")
            return []

        # ADDITIONAL STEP: VGG16 duplicate filtering between candidates
        if debug:
            print(f"\n--- ADDITIONAL: VGG16 duplicate filtering between candidates ---")
        elif not debug:
            animate_step("Removing duplicates", 0.5)
            
        filtered_candidates = []
        candidate_features = []
        
        # Extract features of all candidates
        for i, (vgg16_sim, image_url, source_page_url, img) in enumerate(stage1_candidates):
            features = extract_vgg16_features(img, debug=debug)
            if features is not None:
                candidate_features.append(features)
                filtered_candidates.append((vgg16_sim, image_url, source_page_url, img, features))
        
        # Remove similar images among each other
        unique_candidates = []
        used_indices = set()
        
        for i, (vgg16_sim, image_url, source_page_url, img, features) in enumerate(filtered_candidates):
            if i in used_indices:
                continue
                
            is_unique = True
            for j, (_, _, _, _, other_features) in enumerate(filtered_candidates[:i]):
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
                unique_candidates.append((vgg16_sim, image_url, source_page_url, img))
            else:
                img.close()  # Close similar image
                used_indices.add(i)
        
        if debug:
            print(f"✅ Duplicate filtering: {len(stage1_candidates)} → {len(unique_candidates)} unique candidates")        # STAGE 2: Face-Recognition on ALL filtered candidates
        if debug:
            print(f"\n--- STAGE 2: Face-Recognition (compare all faces at once) ---")
            
        # Extract all face encodings first
        all_face_encodings = []
        valid_candidates = []
        
        total_candidates = len(unique_candidates)
        for i, (vgg16_sim, image_url, source_page_url, img) in enumerate(unique_candidates):
            if not debug:
                show_progress_bar(i + 1, total_candidates, "Stage 2: Face extraction")
                
            # Extract face encoding
            face_encoding = get_face_encoding_from_pil_image(img)
            img.close()
            
            if face_encoding is not None:
                all_face_encodings.append(face_encoding)
                valid_candidates.append((vgg16_sim, image_url, source_page_url))
            # Monitoring
            if debug and (i + 1) % 10 == 0:
                print(f"  Face extraction: {i+1}/{len(unique_candidates)} processed, faces found: {len(all_face_encodings)}")

        if not all_face_encodings:
            if debug:
                print("❌ No faces found in any candidate images!")
            return []

        # Compare ALL faces at once using the official face_recognition approach
        if debug:
            print(f"Comparing {len(all_face_encodings)} faces with query face...")
        
        try:
            # Use face_recognition.compare_faces() exactly like in your example
            results = face_recognition.compare_faces(all_face_encodings, query_face_encoding)
            
            # Also get distances for ranking the matches
            face_distances = face_recognition.face_distance(all_face_encodings, query_face_encoding)            # Collect only the matches (where results[i] == True)
            for i, (is_match, face_distance) in enumerate(zip(results, face_distances)):
                if is_match == True:  # Like: if results[0] == True: print("It's a picture of me!")
                    face_similarity = 1 - face_distance  # Convert distance to similarity
                    
                    # Filter out faces that are not similar (under 50% similarity)
                    if face_similarity < 0.5:
                        if debug:
                            print(f"  Filtering out face {i}: {face_similarity*100:.1f}% similarity (not similar)")
                        continue
                    
                    _, image_url, source_page_url = valid_candidates[i]
                    stage2_matches.append((face_similarity, image_url, source_page_url))
            
            if debug:
                matches_found = len(stage2_matches)
                total_faces = len(results)
                matches_before_filter = len([r for r in results if r == True])
                filtered_faces = matches_before_filter - matches_found
                print(f"✅ Face comparison completed: {matches_before_filter}/{total_faces} faces match, {filtered_faces} filtered out (>50% similarity), {matches_found} kept!")
                
        except Exception as e:
            if debug:
                print(f"❌ Error in face comparison: {e}")
            return []

        # Sort by face similarity (descending order)
        stage2_matches.sort(key=lambda x: x[0], reverse=True)
          # Remove duplicates (same source URLs)
        seen_urls = set()
        unique_matches = []
        for face_similarity, image_url, source_url in stage2_matches:
            if source_url not in seen_urls:
                seen_urls.add(source_url)
                unique_matches.append((face_similarity, image_url, source_url))
        
        if debug and len(stage2_matches) != len(unique_matches):
            print(f"  🔄 {len(stage2_matches) - len(unique_matches)} duplicates removed")
        
        # Limit to top-K
        if len(unique_matches) > top_k:
            unique_matches = unique_matches[:top_k]

        if debug:
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
def show_top_matches(top_matches, debug=False):
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
    failed_images = []  # Track failed image URLs  
    total_matches = len(top_matches)
    
    for i, (similarity, image_url, source_url) in enumerate(top_matches):
        if not debug:
            show_progress_bar(i + 1, total_matches, "Loading thumbnails")
        elif debug:
            print(f"  Loading image {i+1}/{len(top_matches)}: {image_url[:60]}...")
        try:
            resp = requests.get(image_url, timeout=10, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            if resp.status_code != 200:
                if debug:
                    print(f"    ❌ HTTP {resp.status_code}")
                failed_images.append((similarity, image_url, source_url))
                continue
            
            img = Image.open(io.BytesIO(resp.content)).convert('RGB')
            if debug:
                print(f"    ✅ Successfully loaded ({img.size})")
            
        except requests.exceptions.Timeout:
            if debug:
                print(f"    ❌ Timeout while loading")
            failed_images.append((similarity, image_url, source_url))
            continue
        except requests.exceptions.RequestException as e:
            if debug:
                print(f"    ❌ Network error: {e}")
            failed_images.append((similarity, image_url, source_url))
            continue
        except Exception as e:
            if debug:
                print(f"    ❌ Image error: {e}")
            failed_images.append((similarity, image_url, source_url))
            continue
            
        img.thumbnail((200, 200), Image.LANCZOS)
        ax = plt.subplot(rows, cols, i+1)
        ax.imshow(img)
        ax.axis('off')
          # Show similarity and URL
        similarity_percent = similarity * 100
        # Shorten URL for better display (domain + path)
        from urllib.parse import urlparse
        parsed_url = urlparse(source_url)
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
        if debug:
            print("Possible causes:")
            print("- Images are no longer available")
            print("- Servers block access")
            print("- Network problems")
        
        # Show URLs as text fallback when images can't be loaded
        print("\n📋 Showing results as text list since images couldn't be loaded:")
        print("=" * 80)
        for idx, (similarity, image_url, source_url) in enumerate(top_matches, 1):
            similarity_percent = similarity * 100
            print(f"\n🏆 MATCH #{idx}:")
            print(f"   Face similarity: {similarity_percent:.1f}%")
            print(f"   Source page: {source_url}")
            print(f"   Image URL: {image_url}")
            print(f"   {'-' * 60}")
        print("=" * 80)

# ----------------------------------------
# 6. Main Program
# ----------------------------------------
if __name__ == "__main__":
    # 6.1. Adjust parameters
    # Input image path with validation
    while True:
        input_path = input("Path to image: ")
        input_path = input_path.replace('"', '')

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
    face_threshold = 0.7       # No face threshold - sort all faces
    debug = False              # Set to True for detailed debug output
    print(f"=== Face Search Tool with PicImageSearch ===")
    if debug:
        print(f"Using input image: {input_path}")
        print(f"File size: {os.path.getsize(input_path)} bytes")
    else:
        print(f"Analyzing image: {os.path.basename(input_path)}")# 6.2. Extract query features and face encoding
    print("\n=== STEP 1: Analyze query image ===")
    
    # VGG16 features for prefilter
    if not debug:
        animate_step("Loading VGG16 model", 0.5)
    query_features = get_query_features(input_path)
    if query_features is None:
        print("❌ Error extracting VGG16 features from input image!")
        exit(1)
    print("✅ VGG16 features successfully extracted!")
    
    # Face encoding for final face comparisons
    if not debug:
        animate_step("Analyzing face in image", 0.5)
    query_face_encoding = get_face_encoding_from_image(input_path)
    if query_face_encoding is None:
        print("❌ No face detected in input image!")
        exit(1)
    print("✅ Face successfully detected!")    # 6.3. Reverse Image Search (Bing with TinEye fallback)
    print("\n=== STEP 2: Reverse Image Search ===")
    print("Using Bing with TinEye fallback...")
    
    if not debug:
        animate_step("Searching for similar images (Bing + TinEye fallback)", 1.0)
        
    search_start_time = time.time()
    url_pairs = asyncio.run(bing_with_tineye_fallback_search(input_path, max_results=max_results, debug=debug))
    search_total_time = time.time() - search_start_time    
    if debug:
        print(f"\n[MAIN] Reverse image search completed in {search_total_time:.2f}s")
        print(f"[MAIN] Extracted URL pairs: {len(url_pairs)}")
    else:
        print(f"✅ Found {len(url_pairs)} candidate images")
    
    # Analyze search results
    if url_pairs:
        if debug:
            print(f"\n=== URL ANALYSIS ===")
            print(f"First 10 URL pairs:")
            for i, (image_url, source_url) in enumerate(url_pairs[:10], 1):
                print(f"  {i:2d}: Image: {image_url[:50]}...")
                print(f"      Source: {source_url}")
            if len(url_pairs) > 10:
                print(f"  ... and {len(url_pairs)-10} more URL pairs")
            
            # Domain analysis
            from urllib.parse import urlparse
            url_domains = {}
            for image_url, source_url in url_pairs:
                try:
                    domain = urlparse(source_url).netloc
                    url_domains[domain] = url_domains.get(domain, 0) + 1
                except:
                    pass
            
            print(f"\nURL sources:")
            for domain, count in sorted(url_domains.items(), key=lambda x: x[1], reverse=True)[:8]:
                print(f"  {domain}: {count} URLs")
    
    if not url_pairs:
        print("\n❌ NO URLs found via reverse image search (tried Bing and TinEye)!")
        print("Possible causes:")
        print("1. The image is very unusual or not available on the internet")
        print("2. The person is not well-known enough for image databases")
        print("3. Search engines are blocking our requests")
        print("4. The input image has poor quality")
        print("\nRECOMMENDATIONS:")
        print("- Use a different image of the person")
        print("- Try an image with better resolution")
        print("- Use an image of a more well-known person for testing")
        exit(1)    # 6.4. Two-stage processing: VGG16 + Face-Recognition
    print("\n=== STEP 3: Two-stage matching ===")
    print("Using VGG16 prefilter + Face-Recognition...")
    
    if not debug:
        print("This may take a while depending on the number of images found...")
        
    matching_start_time = time.time()
    top_matches = asyncio.run(process_candidate_urls_two_stage(
        url_pairs, 
        query_features, 
        query_face_encoding,
        vgg16_threshold=vgg16_threshold,
        top_k=top_k,
        debug=debug
    ))
    
    matching_total_time = time.time() - matching_start_time
    if debug:
        print(f"\n[MAIN] Two-stage matching completed in {matching_total_time:.2f}s")
    else:
        print(f"✅ Analysis completed in {matching_total_time:.1f}s")

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
        
        for idx, (face_similarity, image_url, source_url) in enumerate(top_matches, 1):
            face_similarity_percent = face_similarity * 100
            print(f"\n🏆 MATCH #{idx}:")
            print(f"   Face similarity: {face_similarity_percent:.1f}%")
            print(f"   Source page: {source_url}")
            print(f"   Image URL: {image_url}")
            print(f"   {'-'*60}")
        
        print(f"\n✅ Best face match: {top_matches[0][0]*100:.1f}% similarity")
        
        if top_matches[0][0] < 0.90:  # Under 90%
            print("⚠️ NOTE: Even with high similarity, these could be different people.")
        else:
            print("✅ Very high face similarity values found!")    # 6.6. Display thumbnails
    print("\n=== STEP 5: Thumbnail Display ===")
    show_top_matches(top_matches, debug=debug)
    
    print("\n✅ Analysis completed!")
