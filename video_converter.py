import streamlit as st
import moviepy.editor as mp
from moviepy.editor import VideoClip, VideoFileClip
from pathlib import Path
import numpy as np
from PIL import Image, ImageFilter
import os
import tempfile
import shutil
import time
import subprocess
import concurrent.futures
from functools import partial
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Set up ffmpeg path - try to use imageio-ffmpeg binary if system ffmpeg is not available
try:
    # Try running ffmpeg to see if it's available
    subprocess.run(["ffmpeg", "-version"], check=True, capture_output=True)
except (subprocess.SubprocessError, FileNotFoundError):
    # If ffmpeg is not found, use imageio-ffmpeg binary
    import imageio_ffmpeg
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    # Update subprocess env path for all calls
    os.environ["PATH"] = os.environ.get("PATH", "") + os.pathsep + str(Path(ffmpeg_path).parent)
    print(f"Using ffmpeg from imageio-ffmpeg: {ffmpeg_path}")

# Check for hardware acceleration support
def check_hw_accel():
    """Check if hardware acceleration is available"""
    try:
        # Check for NVIDIA GPU
        nvidia_result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        if nvidia_result.returncode == 0:
            return "h264_nvenc"  # NVIDIA GPU available
    except:
        pass
    
    try:
        # Check for Intel Quick Sync
        intel_result = subprocess.run(["vainfo"], capture_output=True, text=True)
        if "VAEntrypointEncSlice" in intel_result.stdout:
            return "h264_qsv"  # Intel Quick Sync available
    except:
        pass
    
    return None  # No hardware acceleration available

# Get optimal FFmpeg parameters based on available hardware
def get_ffmpeg_params():
    """Get optimized FFmpeg parameters based on available hardware"""
    hw_accel = check_hw_accel()
    
    if hw_accel == "h264_nvenc":
        return {
            "codec": "h264_nvenc",
            "preset": "p4",  # Faster preset for NVIDIA
            "crf": "23",
            "hwaccel": "cuda",
            "hwaccel_output_format": "cuda"
        }
    elif hw_accel == "h264_qsv":
        return {
            "codec": "h264_qsv",
            "preset": "veryfast",
            "crf": "23",
            "hwaccel": "qsv",
            "hwaccel_output_format": "qsv"
        }
    else:
        return {
            "codec": "libx264",
            "preset": "veryfast",  # Faster preset
            "crf": "23",
            "threads": str(os.cpu_count() or 4)  # Use all available CPU cores
        }

# Process a single video with the given format
def process_video(input_path, output_path, format_type, progress_callback=None):
    """Process a single video with the given format"""
    try:
        logging.info(f"Starting conversion to {format_type} format: {os.path.basename(input_path)}")
        if format_type == "square":
            create_square_video(input_path, output_path)
        elif format_type == "square_blur":
            create_square_blur_video(input_path, output_path)
        elif format_type == "landscape":
            create_landscape_video(input_path, output_path)
        elif format_type == "vertical":
            create_vertical_blur_video(input_path, output_path)
        
        if progress_callback:
            progress_callback()
        
        logging.info(f"Successfully converted to {format_type}: {os.path.basename(output_path)}")
        return True
    except Exception as e:
        logging.error(f"Error processing video {os.path.basename(input_path)}: {str(e)}")
        return False

# Patch moviepy's resize function to use the correct Pillow constant
def patched_resize(clip, newsize=None, height=None, width=None, apply_to_mask=True):
    """
    Patched version of moviepy's resize function that works with newer Pillow versions.
    """
    from PIL import Image
    
    # Determine the target size
    if newsize is not None:
        w, h = newsize
    else:
        w = clip.w if width is None else width
        h = clip.h if height is None else height
    
    # Define the resizer function
    def resizer(pic, newsize):
        # Convert to PIL Image
        pilim = Image.fromarray(pic)
        
        # Use LANCZOS instead of ANTIALIAS for newer Pillow versions
        try:
            # Try the new constant first (Pillow 10.0+)
            resized_pil = pilim.resize(newsize[::-1], Image.Resampling.LANCZOS)
        except AttributeError:
            try:
                # Try the old constant (Pillow 9.x)
                resized_pil = pilim.resize(newsize[::-1], Image.ANTIALIAS)
            except AttributeError:
                # Fallback to default
                resized_pil = pilim.resize(newsize[::-1])
        
        # Convert back to numpy array
        return np.array(resized_pil)
    
    # Apply the resize
    if clip.ismask:
        fl = lambda pic: 1.0*resizer((255 * pic).astype('uint8'), (w, h))
    else:
        fl = lambda pic: resizer(pic.astype('uint8'), (w, h))
    
    newclip = clip.fl_image(fl)
    
    if apply_to_mask and clip.mask is not None:
        newclip.mask = patched_resize(clip.mask, newsize=(w, h), apply_to_mask=False)
    
    return newclip

# Apply the patched resize function to moviepy if needed
# This is a fallback in case the direct resize method has issues
# Fix: Use the correct approach for patching moviepy's resize function
try:
    # Try to patch the resize function directly on the VideoClip class
    VideoClip.resize = patched_resize
except AttributeError:
    # If that fails, we'll use the patched function directly in our code
    pass

def create_square_video(input_path, output_path):
    # Load the video
    video = VideoFileClip(input_path)
    
    # Force video to 30 FPS and calculate adjusted duration
    video = video.set_fps(30)
    frame_duration = 1.0/30  # Duration of one frame at 30fps
    exact_duration = video.duration - (4 * frame_duration)  # Remove 2 frames worth of duration
    
    # Target dimensions (square)
    target_size = 1080
    
    # For portrait videos, we'll crop the center portion
    # Calculate the crop dimensions to maintain aspect ratio
    if video.w > video.h:  # If wider than tall
        crop_width = int(video.h)  # Use height as width
        crop_height = int(video.h)
        x_center = (video.w - crop_width) // 2
        y_center = 0
    else:  # If taller than wide
        crop_width = int(video.w)  # Use width as height
        crop_height = int(video.w)
        x_center = 0
        y_center = (video.h - crop_height) // 2
    
    # Crop the video to square format
    cropped_video = video.crop(x1=x_center, y1=y_center, 
                             x2=x_center + crop_width, 
                             y2=y_center + crop_height)
    
    # Resize to exact target size
    cropped_video = cropped_video.resize((target_size, target_size))
    
    # Set the duration of the final video to match the adjusted duration
    cropped_video = cropped_video.set_duration(exact_duration)
    
    # Write output with high quality settings
    try:
        cropped_video.write_videofile(
            str(output_path),
            codec='libx264',
            audio_codec='aac',
            temp_audiofile='temp-audio.m4a',
            remove_temp=True,
            fps=30,
            bitrate='6000k',  # High bitrate for better quality
            audio_bitrate='320k',  # High audio bitrate
            preset='slow',  # Slower encoding for better quality
            threads=4,  # Use multiple threads for faster processing
            ffmpeg_params=[
                '-profile:v', 'high',  # High profile for better quality
                '-level', '4.1',  # Higher level for better quality
                '-crf', '17',  # Lower CRF value for higher quality (range 0-51, lower is better)
                '-movflags', '+faststart'  # Enable fast start for web playback
            ]
        )
    except Exception as e:
        # Clean up resources
        video.close()
        cropped_video.close()
        if Path(output_path).exists():
            Path(output_path).unlink()
        raise e
    finally:
        # Clean up resources
        video.close()
        cropped_video.close()

def create_square_blur_video_direct(input_path, output_path):
    """Create a square video with blurred background by directly calling ffmpeg."""
    # Get ffmpeg command
    ffmpeg_cmd = get_ffmpeg_path()
    ffprobe_cmd = "ffprobe"  # Also handle ffprobe
    
    try:
        # Create a temporary directory for intermediate files
        temp_dir = tempfile.mkdtemp()
        
        # Paths for intermediate files
        blurred_bg = os.path.join(temp_dir, "blurred_bg.mp4")
        resized_center = os.path.join(temp_dir, "resized_center.mp4")
        audio_file = os.path.join(temp_dir, "audio.aac")
        
        # Check if video has audio stream
        has_audio = False
        try:
            probe_cmd = [ffprobe_cmd, "-v", "error", "-select_streams", "a", 
                        "-show_entries", "stream=codec_type", "-of", "csv=s=x:p=0", 
                        input_path]
            result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
            has_audio = "audio" in result.stdout
        except subprocess.CalledProcessError:
            has_audio = False
        
        # 1. Extract audio if present
        if has_audio:
            try:
                subprocess.run([
                    ffmpeg_cmd, "-i", input_path, "-vn", "-acodec", "copy", 
                    audio_file
                ], check=True, capture_output=True, text=True)
            except subprocess.CalledProcessError as e:
                logging.error(f"Error extracting audio: {e.stderr}")
                raise
        
        # 2. Get video dimensions - with better error handling
        try:
            probe_cmd = [ffprobe_cmd, "-v", "error", "-select_streams", "v:0", 
                       "-show_entries", "stream=width,height", "-of", "csv=s=x:p=0", 
                       input_path]
            result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
            orig_width, orig_height = map(int, result.stdout.strip().split('x'))
        except subprocess.CalledProcessError as e:
            logging.error(f"Error getting video dimensions: {e.stderr}")
            raise
        except ValueError as e:
            logging.error(f"Error parsing video dimensions: {result.stdout}")
            raise
        
        # Target size is 1080x1080
        target_size = 1080
        
        # Calculate dimensions for center video
        # For portrait videos, use full height and calculate width to maintain aspect ratio
        # For landscape videos, use height to fill the square
        if orig_width < orig_height:  # Portrait video
            # Make video take the full height of the square
            visible_height = target_size
            visible_width = int(visible_height * (orig_width / orig_height))
        else:  # Landscape video
            # Make video take the full height of the square
            visible_height = target_size  
            visible_width = int(visible_height * (orig_width / orig_height))
        
        # Ensure width is even (required by H.264)
        visible_width = visible_width if visible_width % 2 == 0 else visible_width + 1
            
        # Calculate position to center horizontally
        x_offset = (target_size - visible_width) // 2
        y_offset = 0  # No vertical offset since height is full
        
        # 3. Create blurred background - with better error handling
        try:
            subprocess.run([
                ffmpeg_cmd, "-i", input_path, "-vf", 
                "scale=1080:1080:force_original_aspect_ratio=increase,crop=1080:1080,boxblur=30:5", 
                "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "23", 
                blurred_bg
            ], check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            logging.error(f"Error creating blurred background: {e.stderr}")
            raise
        
        # 4. Create centered video - with better error handling
        try:
            subprocess.run([
                ffmpeg_cmd, "-i", input_path, "-vf", 
                f"scale={visible_width}:-2", 
                "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "23", 
                resized_center
            ], check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            logging.error(f"Error creating centered video: {e.stderr}")
            raise
        
        # 5. Overlay centered video on blurred background - with better error handling
        try:
            if has_audio:
                subprocess.run([
                    ffmpeg_cmd, "-i", blurred_bg, "-i", resized_center, "-i", audio_file,
                    "-filter_complex", f"[0:v][1:v] overlay={x_offset}:{y_offset} [outv]", 
                    "-map", "[outv]", "-map", "2:a", "-c:v", "libx264", "-c:a", "aac",
                    "-shortest", output_path
                ], check=True, capture_output=True, text=True)
            else:
                subprocess.run([
                    ffmpeg_cmd, "-i", blurred_bg, "-i", resized_center,
                    "-filter_complex", f"[0:v][1:v] overlay={x_offset}:{y_offset} [outv]", 
                    "-map", "[outv]", "-c:v", "libx264",
                    "-shortest", output_path
                ], check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            logging.error(f"Error overlaying videos: {e.stderr}")
            raise
        
    except subprocess.CalledProcessError as e:
        # Clean up and raise an error with more details
        error_msg = f"FFmpeg error: {e.stderr if hasattr(e, 'stderr') else 'Unknown error'}"
        logging.error(error_msg)
        if Path(output_path).exists():
            Path(output_path).unlink()
        raise Exception(error_msg)
    except Exception as e:
        # Handle other exceptions
        logging.error(f"Unexpected error: {str(e)}")
        if Path(output_path).exists():
            Path(output_path).unlink()
        raise
    finally:
        # Clean up temp files
        for file in [blurred_bg, resized_center, audio_file]:
            if Path(file).exists():
                Path(file).unlink()
        if Path(temp_dir).exists():
            Path(temp_dir).rmdir()

def create_square_blur_video(input_path, output_path):
    """
    Create a square video with blurred background (wrapper for the direct implementation).
    This function maintains backwards compatibility.
    """
    # Just call the direct implementation
    create_square_blur_video_direct(input_path, output_path)

def create_landscape_video_direct(input_path, output_path):
    """Create a landscape video by directly calling ffmpeg."""
    # Get ffmpeg command
    ffmpeg_cmd = get_ffmpeg_path()
    ffprobe_cmd = "ffprobe"  # Also handle ffprobe
    
    try:
        # Create a temporary directory for intermediate files
        temp_dir = tempfile.mkdtemp()
        
        # Paths for intermediate files
        blurred_bg = os.path.join(temp_dir, "blurred_bg.mp4")
        resized_center = os.path.join(temp_dir, "resized_center.mp4")
        audio_file = os.path.join(temp_dir, "audio.aac")
        
        # Check if video has audio stream
        has_audio = False
        try:
            probe_cmd = [ffprobe_cmd, "-v", "error", "-select_streams", "a", 
                        "-show_entries", "stream=codec_type", "-of", "csv=s=x:p=0", 
                        input_path]
            result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
            has_audio = "audio" in result.stdout
        except subprocess.CalledProcessError:
            has_audio = False
        
        # 1. Extract audio if present
        if has_audio:
            try:
                subprocess.run([
                    ffmpeg_cmd, "-i", input_path, "-vn", "-acodec", "copy", 
                    audio_file
                ], check=True, capture_output=True, text=True)
            except subprocess.CalledProcessError as e:
                logging.error(f"Error extracting audio: {e.stderr}")
                raise
        
        # 2. Get video dimensions - with better error handling
        try:
            probe_cmd = [ffprobe_cmd, "-v", "error", "-select_streams", "v:0", 
                       "-show_entries", "stream=width,height", "-of", "csv=s=x:p=0", 
                       input_path]
            result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
            orig_width, orig_height = map(int, result.stdout.strip().split('x'))
        except subprocess.CalledProcessError as e:
            logging.error(f"Error getting video dimensions: {e.stderr}")
            raise
        except ValueError as e:
            logging.error(f"Error parsing video dimensions: {result.stdout}")
            raise
        
        # Calculate target dimensions ensuring they're even
        target_height = 1080
        # Calculate width while maintaining aspect ratio
        target_width = int((orig_width / orig_height) * target_height)
        # Ensure width is even
        target_width = target_width if target_width % 2 == 0 else target_width + 1
        
        # 3. Create blurred background - with better error handling
        try:
            subprocess.run([
                ffmpeg_cmd, "-i", input_path, "-vf", 
                f"scale=1920:1080:force_original_aspect_ratio=increase,crop=1920:1080,boxblur=20:5", 
                "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "23", 
                blurred_bg
            ], check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            logging.error(f"Error creating blurred background: {e.stderr}")
            raise
        
        # 4. Create centered video - with better error handling
        try:
            subprocess.run([
                ffmpeg_cmd, "-i", input_path, "-vf", 
                f"scale={target_width}:{target_height}", 
                "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "23", 
                resized_center
            ], check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            logging.error(f"Error creating centered video: {e.stderr}")
            raise
        
        # 5. Overlay centered video on blurred background
        # Calculate position for centered overlay
        x_offset = (1920 - target_width) // 2
        
        # Composite videos - with better error handling
        try:
            if has_audio:
                subprocess.run([
                    ffmpeg_cmd, "-i", blurred_bg, "-i", resized_center, "-i", audio_file,
                    "-filter_complex", f"[0:v][1:v] overlay={x_offset}:0 [outv]", 
                    "-map", "[outv]", "-map", "2:a", "-c:v", "libx264", "-c:a", "aac",
                    "-shortest", output_path
                ], check=True, capture_output=True, text=True)
            else:
                subprocess.run([
                    ffmpeg_cmd, "-i", blurred_bg, "-i", resized_center,
                    "-filter_complex", f"[0:v][1:v] overlay={x_offset}:0 [outv]", 
                    "-map", "[outv]", "-c:v", "libx264",
                    "-shortest", output_path
                ], check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            logging.error(f"Error overlaying videos: {e.stderr}")
            raise
        
    except subprocess.CalledProcessError as e:
        # Clean up and raise an error with more details
        error_msg = f"FFmpeg error: {e.stderr if hasattr(e, 'stderr') else 'Unknown error'}"
        logging.error(error_msg)
        if Path(output_path).exists():
            Path(output_path).unlink()
        raise Exception(error_msg)
    except Exception as e:
        # Handle other exceptions
        logging.error(f"Unexpected error: {str(e)}")
        if Path(output_path).exists():
            Path(output_path).unlink()
        raise
    finally:
        # Clean up temp files
        for file in [blurred_bg, resized_center, audio_file]:
            if Path(file).exists():
                Path(file).unlink()
        if Path(temp_dir).exists():
            Path(temp_dir).rmdir()

def create_landscape_video(input_path, output_path):
    """
    Create a landscape video (wrapper for the direct implementation).
    This function maintains backwards compatibility.
    """
    # Just call the direct implementation
    create_landscape_video_direct(input_path, output_path)

def get_video_metadata(video_path):
    """Get metadata for a video file"""
    try:
        video = VideoFileClip(video_path)
        duration = video.duration
        size_mb = os.path.getsize(video_path) / (1024 * 1024)
        video.close()
        return {
            "duration": f"{duration:.2f} seconds",
            "size": f"{size_mb:.2f} MB"
        }
    except Exception as e:
        return {
            "duration": "Unknown",
            "size": "Unknown"
        }

# Define a helper function to get the ffmpeg binary path
def get_ffmpeg_path():
    """Get the ffmpeg binary path, either from moviepy's config or our environment variable."""
    from moviepy.config import get_setting
    try:
        return get_setting("FFMPEG_BINARY")
    except:
        # Fallback to direct command
        return "ffmpeg"

def create_vertical_blur_video_direct(input_path, output_path):
    """Create a vertical video with blurred background by directly calling ffmpeg."""
    # Get ffmpeg command
    ffmpeg_cmd = get_ffmpeg_path()
    ffprobe_cmd = "ffprobe"  # Also handle ffprobe
    
    try:
        # Create a temporary directory for intermediate files
        temp_dir = tempfile.mkdtemp()
        
        # Paths for intermediate files
        blurred_bg = os.path.join(temp_dir, "blurred_bg.mp4")
        resized_center = os.path.join(temp_dir, "resized_center.mp4")
        audio_file = os.path.join(temp_dir, "audio.aac")
        
        # Check if video has audio stream
        has_audio = False
        try:
            probe_cmd = [ffprobe_cmd, "-v", "error", "-select_streams", "a", 
                        "-show_entries", "stream=codec_type", "-of", "csv=s=x:p=0", 
                        input_path]
            result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
            has_audio = "audio" in result.stdout
        except subprocess.CalledProcessError:
            has_audio = False
        
        # 1. Extract audio if present
        if has_audio:
            try:
                subprocess.run([
                    ffmpeg_cmd, "-i", input_path, "-vn", "-acodec", "copy", 
                    audio_file
                ], check=True, capture_output=True, text=True)
            except subprocess.CalledProcessError as e:
                logging.error(f"Error extracting audio: {e.stderr}")
                raise
        
        # 2. Get video dimensions - with better error handling
        try:
            probe_cmd = [ffprobe_cmd, "-v", "error", "-select_streams", "v:0", 
                       "-show_entries", "stream=width,height", "-of", "csv=s=x:p=0", 
                       input_path]
            result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
            orig_width, orig_height = map(int, result.stdout.strip().split('x'))
        except subprocess.CalledProcessError as e:
            logging.error(f"Error getting video dimensions: {e.stderr}")
            raise
        except ValueError as e:
            logging.error(f"Error parsing video dimensions: {result.stdout}")
            raise
        
        # Target dimensions
        target_width = 1080
        target_height = 1920
        
        # Calculate dimensions for center video
        # For square videos, we'll scale to fit the width
        # For landscape videos, we'll scale to fit the height
        # For portrait videos, we'll scale to fit the width
        if orig_width == orig_height:  # Square video
            visible_width = target_width
            visible_height = int(visible_width * (orig_height / orig_width))
        elif orig_width > orig_height:  # Landscape video
            visible_height = target_height
            visible_width = int(visible_height * (orig_width / orig_height))
        else:  # Portrait video
            visible_width = target_width
            visible_height = int(visible_width * (orig_height / orig_width))
        
        # Ensure dimensions are even (required by H.264)
        visible_width = visible_width if visible_width % 2 == 0 else visible_width + 1
        visible_height = visible_height if visible_height % 2 == 0 else visible_height + 1
        
        # Calculate position to center the video
        x_offset = (target_width - visible_width) // 2
        y_offset = (target_height - visible_height) // 2
        
        # 3. Create blurred background - with better error handling
        try:
            subprocess.run([
                ffmpeg_cmd, "-i", input_path, "-vf", 
                f"scale={target_width}:{target_height}:force_original_aspect_ratio=increase,crop={target_width}:{target_height},boxblur=30:5", 
                "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "23", 
                blurred_bg
            ], check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            logging.error(f"Error creating blurred background: {e.stderr}")
            raise
        
        # 4. Create centered video - with better error handling
        try:
            subprocess.run([
                ffmpeg_cmd, "-i", input_path, "-vf", 
                f"scale={visible_width}:{visible_height}", 
                "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "23", 
                resized_center
            ], check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            logging.error(f"Error creating centered video: {e.stderr}")
            raise
        
        # 5. Overlay centered video on blurred background - with better error handling
        try:
            if has_audio:
                subprocess.run([
                    ffmpeg_cmd, "-i", blurred_bg, "-i", resized_center, "-i", audio_file,
                    "-filter_complex", f"[0:v][1:v] overlay={x_offset}:{y_offset} [outv]", 
                    "-map", "[outv]", "-map", "2:a", "-c:v", "libx264", "-c:a", "aac",
                    "-shortest", output_path
                ], check=True, capture_output=True, text=True)
            else:
                subprocess.run([
                    ffmpeg_cmd, "-i", blurred_bg, "-i", resized_center,
                    "-filter_complex", f"[0:v][1:v] overlay={x_offset}:{y_offset} [outv]", 
                    "-map", "[outv]", "-c:v", "libx264",
                    "-shortest", output_path
                ], check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            logging.error(f"Error overlaying videos: {e.stderr}")
            raise
        
    except subprocess.CalledProcessError as e:
        # Clean up and raise an error with more details
        error_msg = f"FFmpeg error: {e.stderr if hasattr(e, 'stderr') else 'Unknown error'}"
        logging.error(error_msg)
        if Path(output_path).exists():
            Path(output_path).unlink()
        raise Exception(error_msg)
    except Exception as e:
        # Handle other exceptions
        logging.error(f"Unexpected error: {str(e)}")
        if Path(output_path).exists():
            Path(output_path).unlink()
        raise
    finally:
        # Clean up temp files
        for file in [blurred_bg, resized_center, audio_file]:
            if Path(file).exists():
                Path(file).unlink()
        if Path(temp_dir).exists():
            Path(temp_dir).rmdir()

def create_vertical_blur_video(input_path, output_path):
    """
    Create a vertical video with blurred background (wrapper for the direct implementation).
    This function maintains backwards compatibility.
    """
    # Just call the direct implementation
    create_vertical_blur_video_direct(input_path, output_path)

def main():
    # Set page config with a nice title and icon
    st.set_page_config(
        page_title="Photoroom Video Format Converter",
        page_icon="🎥",
        layout="wide"
    )

    st.markdown("""
    Made with ❤️ by Jiali
    """)
    
    # Main title with emoji
    st.title("🎥 Photoroom Video Format Converter")
    
    # Initialize stop flag in session state if it doesn't exist
    if 'stop_processing' not in st.session_state:
        st.session_state.stop_processing = False
    
    # Initialize processed videos list in session state if it doesn't exist
    if 'processed_videos' not in st.session_state:
        st.session_state.processed_videos = []
    
    # Add custom output directory option
    st.subheader("Output Settings")
    use_custom_output = st.checkbox("Use custom output directory", value=False)
    
    if use_custom_output:
        st.info("💡 Click the button below to select where to save your converted videos:")
        
        # Initialize session state for selected directory
        if 'selected_output_dir' not in st.session_state:
            st.session_state.selected_output_dir = os.path.expanduser("~/Downloads")
        
        # Create folder picker using HTML5 and JavaScript
        import streamlit.components.v1 as components
        
        folder_picker_html = """
        <div style="margin: 10px 0;">
            <button id="folderPicker" style="
                background: linear-gradient(90deg, #ff6b6b, #ee5a24);
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 8px;
                cursor: pointer;
                font-size: 16px;
                font-weight: bold;
                transition: all 0.3s ease;
                box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
            " onmouseover="this.style.transform='translateY(-2px)'; this.style.boxShadow='0 6px 20px rgba(0, 0, 0, 0.3)';" 
               onmouseout="this.style.transform='translateY(0px)'; this.style.boxShadow='0 4px 15px rgba(0, 0, 0, 0.2)';">
                📁 Select Output Folder
            </button>
            <input type="file" id="folderInput" webkitdirectory style="display: none;">
            <div id="selectedPath" style="margin-top: 10px; padding: 10px; background: #f0f2f6; border-radius: 5px; display: none;">
                <strong>Selected folder:</strong> <span id="pathText"></span>
            </div>
        </div>
        
        <script>
        document.getElementById('folderPicker').onclick = function() {
            document.getElementById('folderInput').click();
        };
        
        document.getElementById('folderInput').onchange = function(e) {
            if (e.target.files.length > 0) {
                var path = e.target.files[0].webkitRelativePath;
                var folderPath = path.substring(0, path.lastIndexOf('/'));
                
                document.getElementById('pathText').textContent = folderPath || 'Root folder selected';
                document.getElementById('selectedPath').style.display = 'block';
                
                // Send the folder path to Streamlit
                var event = new CustomEvent('folderSelected', {
                    detail: { folderPath: folderPath }
                });
                window.parent.document.dispatchEvent(event);
            }
        };
        </script>
        """
        
        # Show the folder picker
        components.html(folder_picker_html, height=120)
        
        # Alternative: Simple text input with common suggestions
        st.markdown("**Or enter the path manually:**")
        col1, col2 = st.columns([3, 1])
        
        with col1:
            custom_path = st.text_input(
                "Output directory path:",
                value=st.session_state.selected_output_dir,
                placeholder="e.g., /Users/yourname/Downloads/MyVideos"
            )
        
        with col2:
            if st.button("📂 Use Common Folders"):
                # Show quick options
                st.session_state.show_quick_options = True
        
        # Quick folder options
        if hasattr(st.session_state, 'show_quick_options') and st.session_state.show_quick_options:
            st.markdown("**Quick options:**")
            quick_cols = st.columns(4)
            
            common_dirs = [
                ("Downloads", os.path.expanduser("~/Downloads")),
                ("Desktop", os.path.expanduser("~/Desktop")),
                ("Documents", os.path.expanduser("~/Documents")),
                ("Current Dir", os.getcwd())
            ]
            
            for i, (name, path) in enumerate(common_dirs):
                with quick_cols[i % 4]:
                    if st.button(f"📁 {name}", key=f"quick_{i}"):
                        if os.path.exists(path):
                            st.session_state.selected_output_dir = path
                            custom_path = path
                            st.session_state.show_quick_options = False
                            st.rerun()
                        else:
                            st.error(f"{name} folder not found!")
        
        # Update the selected directory
        if custom_path:
            st.session_state.selected_output_dir = custom_path
            output_dir = custom_path
            
            # Validate directory
            if not os.path.exists(output_dir):
                st.warning(f"⚠️ Directory does not exist: `{output_dir}`")
                if st.button("📁 Create Directory", type="primary"):
                    try:
                        os.makedirs(output_dir, exist_ok=True)
                        st.success(f"✅ Created directory: `{output_dir}`")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Failed to create directory: {str(e)}")
                return
            else:
                st.success(f"✅ Videos will be saved to: `{output_dir}`")
        else:
            output_dir = st.session_state.selected_output_dir
            
    else:
        # Create output directory in the current working directory
        output_dir = os.path.join(os.getcwd(), "converted_videos")
        os.makedirs(output_dir, exist_ok=True)
    
    logging.info(f"Output directory set to: {output_dir}")
    
    # Add format selection with checkboxes
    st.subheader("Select output formats")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        square_format = st.checkbox("Square (1080x1080)", value=True)
    with col2:
        square_blur_format = st.checkbox("Square with Blur (1080x1080)", value=False)
    with col3:
        landscape_format = st.checkbox("Landscape with Blur (1920x1080)", value=False)
    with col4:
        vertical_blur_format = st.checkbox("Vertical (1080x1920)", value=False)
    
    # File uploader for multiple videos
    uploaded_files = st.file_uploader("Upload videos", type=['mp4', 'mov'], accept_multiple_files=True)
    
    if uploaded_files:
        logging.info(f"Received {len(uploaded_files)} videos for processing")
        
        # Create two columns for Convert and Stop buttons
        col1, col2 = st.columns([4, 1])
        
        # Add large convert button
        if col1.button("🚀 Convert Videos!", type="primary", use_container_width=True):
            st.session_state.stop_processing = False
            
            # Add loading message
            st.info("⏳ Video conversion is in progress. Your video's loading… almost showtime! 🍿")
            
            progress_text = st.empty()
            progress_bar = st.progress(0)
            
            # Clear previous processed videos
            st.session_state.processed_videos = []
            
            # Create a temporary directory only for input files
            temp_dir = tempfile.mkdtemp()
            logging.info(f"Created temporary directory: {temp_dir}")
            
            # Prepare all conversion tasks
            conversion_tasks = []
            for i, uploaded_file in enumerate(uploaded_files):
                input_path = os.path.join(temp_dir, f"temp_input_{i}.mp4")
                with open(input_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                
                if square_format:
                    output_filename = f"{os.path.splitext(uploaded_file.name)[0]}_square.mp4"
                    output_path = os.path.join(output_dir, output_filename)
                    conversion_tasks.append((input_path, output_path, "square", output_filename, uploaded_file.name, "Square (1080x1080)"))
                
                if square_blur_format:
                    output_filename = f"{os.path.splitext(uploaded_file.name)[0]}_square_blur.mp4"
                    output_path = os.path.join(output_dir, output_filename)
                    conversion_tasks.append((input_path, output_path, "square_blur", output_filename, uploaded_file.name, "Square with Blur (1080x1080)"))
                
                if landscape_format:
                    output_filename = f"{os.path.splitext(uploaded_file.name)[0]}_landscape.mp4"
                    output_path = os.path.join(output_dir, output_filename)
                    conversion_tasks.append((input_path, output_path, "landscape", output_filename, uploaded_file.name, "Landscape (1920x1080)"))
                
                if vertical_blur_format:
                    output_filename = f"{os.path.splitext(uploaded_file.name)[0]}_vertical.mp4"
                    output_path = os.path.join(output_dir, output_filename)
                    conversion_tasks.append((input_path, output_path, "vertical", output_filename, uploaded_file.name, "Vertical (1080x1920)"))
            
            total_conversions = len(conversion_tasks)
            
            logging.info(f"Starting conversion of {total_conversions} videos")
            
            # Process videos in parallel using ThreadPoolExecutor
            failed_conversions = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(os.cpu_count() or 4, total_conversions)) as executor:
                futures = []
                for input_path, output_path, format_type, output_filename, original_name, format_name in conversion_tasks:
                    if st.session_state.stop_processing:
                        logging.warning("Processing stopped by user")
                        break
                    
                    progress_text.text(f"Processing {original_name} to {format_name}...")
                    future = executor.submit(
                        process_video,
                        input_path,
                        output_path,
                        format_type,
                        None  # Remove progress callback to avoid threading issues
                    )
                    futures.append((future, output_filename, output_path, original_name, format_name))
                
                # Wait for all tasks to complete and update progress in main thread
                for i, (future, output_filename, output_path, original_name, format_name) in enumerate(futures):
                    try:
                        if future.result():
                            st.session_state.processed_videos.append({
                                "name": output_filename,
                                "path": output_path,
                                "original_name": original_name,
                                "format": format_name
                            })
                        else:
                            failed_conversions.append(f"{original_name} to {format_name}")
                    except Exception as e:
                        failed_conversions.append(f"{original_name} to {format_name}: {str(e)}")
                        logging.error(f"Exception in future result: {str(e)}")
                    
                    # Update progress in main thread
                    current_conversion = i + 1
                    progress = current_conversion / total_conversions
                    progress_bar.progress(progress)
                    remaining = total_conversions - current_conversion
                    logging.info(f"Progress: {current_conversion}/{total_conversions} videos processed. {remaining} videos remaining.")
            
            # Clean up temp directory
            shutil.rmtree(temp_dir)
            logging.info("Cleaned up temporary directory")
            
            if not st.session_state.stop_processing:
                progress_text.text("All videos processed!")
                logging.info("All videos processed successfully")
                
                # Display success message and any failures
                if failed_conversions:
                    st.warning(f"⚠️ Some conversions failed. Check the logs for details.")
                    with st.expander("Failed conversions"):
                        for failure in failed_conversions:
                            st.text(f"❌ {failure}")
                    if st.session_state.processed_videos:
                        st.success(f"✨ Some videos have been converted successfully! Saved in: {output_dir}")
                else:
                    st.success(f"✨ All videos have been converted successfully! Saved in: {output_dir}")
        
        # Add stop button
        if col2.button("⏹️ Stop", type="secondary", use_container_width=True):
            st.session_state.stop_processing = True
            logging.warning("Stop button pressed - stopping processing")
    
    # Display preview and download section if there are processed videos
    if st.session_state.processed_videos:
        st.subheader("🎬 Preview and Download")
        st.write("Your converted videos are ready! Preview and download them below.")
        
        # Add a clear button to reset the processed videos
        if st.button("🗑️ Clear All Videos", type="secondary"):
            st.session_state.processed_videos = []
            st.rerun()
        
        # Add batch download option - show even with a single video
        st.info("💡 Tip: You can download individual videos or use the batch download option below.")
        
        # Create a zip file with all videos for batch download
        if st.button("📦 Zip all videos", type="primary", use_container_width=True):
            with st.spinner("Preparing ZIP file..."):
                # Create a temporary zip file
                zip_path = os.path.join(tempfile.gettempdir(), "converted_videos.zip")
                
                # Create a zip file with all videos
                import zipfile
                with zipfile.ZipFile(zip_path, 'w') as zipf:
                    for video in st.session_state.processed_videos:
                        zipf.write(video["path"], arcname=video["name"])
                
                # Provide download button for the zip file
                with open(zip_path, "rb") as f:
                    st.download_button(
                        label="📥 Download ZIP File",
                        data=f,
                        file_name="converted_videos.zip",
                        mime="application/zip",
                        use_container_width=True
                    )
        
        # Group videos by original file
        original_files = {}
        for video in st.session_state.processed_videos:
            if video["original_name"] not in original_files:
                original_files[video["original_name"]] = []
            original_files[video["original_name"]].append(video)
        
        # Create a container with a smaller width for the video display
        with st.container():
            # Display each original file's videos in a separate section
            for original_name, videos in original_files.items():
                # Create a card-like container for each original file
                with st.expander(f"📹 Original: {original_name}", expanded=True):
                    # Create columns for the videos
                    cols = st.columns(min(len(videos), 3))
                    
                    # Display each video in a column
                    for i, video in enumerate(videos):
                        with cols[i % len(cols)]:
                            st.write(f"**{video['format']}**")
                            
                            # Get video metadata
                            metadata = get_video_metadata(video["path"])
                            
                            # Only show video preview if total number of files is 10 or less
                            if len(st.session_state.processed_videos) <= 10:
                                # Display video preview with a smaller size
                                st.video(video["path"], start_time=0)
                            
                            # Display metadata
                            st.caption(f"⏱️ Duration: {metadata['duration']}")
                            st.caption(f"📊 Size: {metadata['size']}")
                            
                            # Add download button
                            with open(video["path"], "rb") as file:
                                st.download_button(
                                    label=f"📥 Download",
                                    data=file,
                                    file_name=video["name"],
                                    mime="video/mp4",
                                    key=f"download_{video['name']}",
                                    use_container_width=True
                                )
                
                # Add a divider between different original files
                st.divider()
    
    st.markdown("### Format examples")
    
    # Display example image with smaller size
    st.image("exemple.png", caption="Format examples", width=600)

if __name__ == "__main__":
    main() 