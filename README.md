# Face Search Tool 🔍👤

A powerful automated reverse image search tool that combines Bing and TinEye image search capabilities with advanced face recognition technology to find similar faces across the internet.

## Features ✨

- **Automated Reverse Image Search**: Uses PicImageSearch to query Bing and TinEye (fallback) without requiring API keys
- **Two-Stage Filtering System**:
  - **Stage 1**: VGG16-based image similarity filtering to remove duplicates and too-similar images
  - **Stage 2**: Advanced face recognition using 128D face encodings for precise face matching
- **Smart Progress Indicators**: Visual progress bars and loading animations for better user experience
- **Thumbnail Gallery**: Visual display of top matches with similarity scores
- **Duplicate Detection**: Intelligent removal of duplicate results

## How It Works 🔧

1. **Face Analysis**: Extracts face encoding from your input image using face_recognition library
2. **Image Search**: Searches Bing for visually similar images using PicImageSearch (with TinEye as fallback if no results)
3. **VGG16 Filtering**: Removes images that are too similar to the original (likely duplicates)
4. **Face Matching**: Analyzes faces in remaining images and ranks by similarity
5. **Results Display**: Shows top matches with similarity scores and thumbnail gallery

## Installation 📦

Clone this repository with:

```bash
git clone https://github.com/Hawk3388/face-search
cd face-search
```

Install all the requirements:

```bash
pip install -r requirements.txt
```

## Usage 🚀

### Basic Usage

```bash
python main.py
```

The program will prompt you for an image path:

```bash
Path to image: <image-path>
```

### Supported Image Formats 📸

- `.jpg`, `.jpeg`
- `.png`
- `.gif`
- `.bmp`
- `.webp`
- `.tiff`

## Technical Details 🔬

### Dependencies

- **Python 3.7+**: Required for all libraries

### Algorithm Overview

The tool uses a sophisticated two-stage filtering approach:

1. **VGG16 Prefiltering**: Uses deep learning features to quickly eliminate very similar or duplicate images
2. **Face Recognition**: Applies the state-of-the-art face_recognition library to compare 128-dimensional face encodings

### Performance

- Processes 100+ candidate images in typically 10-30 seconds
- Memory efficient with streaming image processing
- Concurrent downloads for faster processing

## Privacy & Ethics ⚠️

This tool is designed for legitimate research and educational purposes. Please:

- Respect privacy and obtain proper consent when searching for people
- Follow applicable laws and regulations in your jurisdiction
- Use responsibly and ethically
- Be aware that results may include false positives
- The program uses Bing and TinEye reverse image search, so be aware that images you upload are processed by these services

## Project Structure 📁

```bash
face-search/
├── main.py              # Main application
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

## Contributing 🤝

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Use of AI

This project was built with AI, including the generation of code and code snippets and parts of this README file.

## License 📄

This project is for educational and research purposes. Please ensure compliance with all applicable laws and terms of service of the search engines used.

## Credits 🙏

- **face_recognition**: Adam Geitgey's face recognition library
- **PicImageSearch**: Reverse image search implementation for Bing and TinEye
- **TensorFlow**: VGG16 model for feature extraction
