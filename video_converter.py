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

# Patch moviepy's resize function to use the correct Pillow constant
from functools import partial

# Define a patched version of the resize function
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
    # Import necessary modules
    import subprocess
    import os
    import tempfile
    from pathlib import Path
    
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
        
        # 1. Extract audio
        subprocess.run([
            ffmpeg_cmd, "-i", input_path, "-vn", "-acodec", "copy", 
            audio_file
        ], check=True, capture_output=True)
        
        # 2. Get video dimensions
        probe_cmd = [ffprobe_cmd, "-v", "error", "-select_streams", "v:0", 
                   "-show_entries", "stream=width,height", "-of", "csv=s=x:p=0", 
                   input_path]
        result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
        orig_width, orig_height = map(int, result.stdout.strip().split('x'))
        
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
        
        # 3. Create blurred background (scale to fill 1080x1080, then blur)
        # Scale to fill while maintaining aspect ratio, then crop to square
        subprocess.run([
            ffmpeg_cmd, "-i", input_path, "-vf", 
            "scale=1080:1080:force_original_aspect_ratio=increase,crop=1080:1080,boxblur=30:5", 
            "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "23", 
            blurred_bg
        ], check=True, capture_output=True)
        
        # 4. Create centered video using a safer filter approach
        # Use the scale filter with -2 to ensure divisible by 2 (required for h264)
        subprocess.run([
            ffmpeg_cmd, "-i", input_path, "-vf", 
            f"scale={visible_width}:-2", 
            "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "23", 
            resized_center
        ], check=True, capture_output=True)
        
        # 5. Overlay centered video on blurred background
        subprocess.run([
            ffmpeg_cmd, "-i", blurred_bg, "-i", resized_center, "-i", audio_file,
            "-filter_complex", f"[0:v][1:v] overlay={x_offset}:{y_offset} [outv]", 
            "-map", "[outv]", "-map", "2:a", "-c:v", "libx264", "-c:a", "aac",
            "-shortest", output_path
        ], check=True, capture_output=True)
        
    except subprocess.CalledProcessError as e:
        # Clean up and raise an error
        print(f"Error: {e}")
        print(f"STDOUT: {e.stdout.decode() if e.stdout else 'None'}")
        print(f"STDERR: {e.stderr.decode() if e.stderr else 'None'}")
        if Path(output_path).exists():
            Path(output_path).unlink()
        raise e
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
        
        # 1. Extract audio
        subprocess.run([
            ffmpeg_cmd, "-i", input_path, "-vn", "-acodec", "copy", 
            audio_file
        ], check=True, capture_output=True)
        
        # 2. Create blurred background (scale to fill 1920x1080, then blur)
        subprocess.run([
            ffmpeg_cmd, "-i", input_path, "-vf", 
            "scale=1920:1080:force_original_aspect_ratio=increase,crop=1920:1080,boxblur=20:5", 
            "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "23", 
            blurred_bg
        ], check=True, capture_output=True)
        
        # 3. Create centered video (scale to height=1080, maintain aspect ratio)
        subprocess.run([
            ffmpeg_cmd, "-i", input_path, "-vf", 
            "scale=-1:1080", 
            "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "23", 
            resized_center
        ], check=True, capture_output=True)
        
        # 4. Overlay centered video on blurred background
        # First get dimensions of the centered video
        probe_cmd = [ffprobe_cmd, "-v", "error", "-select_streams", "v:0", 
                    "-show_entries", "stream=width,height", "-of", "csv=s=x:p=0", 
                    resized_center]
        result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
        width, height = map(int, result.stdout.strip().split('x'))
        
        # Calculate position for centered overlay
        x_offset = (1920 - width) // 2
        
        # Composite videos
        subprocess.run([
            ffmpeg_cmd, "-i", blurred_bg, "-i", resized_center, "-i", audio_file,
            "-filter_complex", f"[0:v][1:v] overlay={x_offset}:0 [outv]", 
            "-map", "[outv]", "-map", "2:a", "-c:v", "libx264", "-c:a", "aac",
            "-shortest", output_path
        ], check=True, capture_output=True)
        
    except subprocess.CalledProcessError as e:
        # Clean up and raise an error
        print(f"Error: {e}")
        print(f"STDOUT: {e.stdout.decode() if e.stdout else 'None'}")
        print(f"STDERR: {e.stderr.decode() if e.stderr else 'None'}")
        if Path(output_path).exists():
            Path(output_path).unlink()
        raise e
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

def main():
    # Set page config with a nice title and icon
    st.set_page_config(
        page_title="Photoroom Video Format Converter",
        page_icon="🎥",
        layout="wide"
    )
    
    # Main title with emoji
    st.title("🎥 Photoroom Video Format Converter")
    
    # Initialize stop flag in session state if it doesn't exist
    if 'stop_processing' not in st.session_state:
        st.session_state.stop_processing = False
    
    # Initialize processed videos list in session state if it doesn't exist
    if 'processed_videos' not in st.session_state:
        st.session_state.processed_videos = []
    
    # Add format selection with checkboxes
    st.subheader("Select output formats")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        square_format = st.checkbox("Square (1080x1080)", value=True)
    with col2:
        square_blur_format = st.checkbox("Square with Blur (1080x1080)", value=False)
    with col3:
        landscape_format = st.checkbox("Landscape (1920x1080)", value=False)
    
    # File uploader for multiple videos
    uploaded_files = st.file_uploader("Upload videos", type=['mp4', 'mov'], accept_multiple_files=True)
    
    if uploaded_files:
        # Create two columns for Convert and Stop buttons
        col1, col2 = st.columns([4, 1])
        
        # Add large convert button
        if col1.button("🚀 Convert Videos!", type="primary", use_container_width=True):
            st.session_state.stop_processing = False
            progress_text = st.empty()
            progress_bar = st.progress(0)
            
            # Clear previous processed videos
            st.session_state.processed_videos = []
            
            # Create a temporary directory to store processed videos
            temp_dir = tempfile.mkdtemp()
            
            total_conversions = len(uploaded_files) * (square_format + square_blur_format + landscape_format)
            current_conversion = 0
            
            for i, uploaded_file in enumerate(uploaded_files):
                # Check if stop button was clicked
                if st.session_state.stop_processing:
                    progress_text.text("Processing cancelled!")
                    st.warning("Video conversion was stopped.")
                    break
                
                # Create temp input file
                input_path = os.path.join(temp_dir, f"temp_input_{i}.mp4")
                with open(input_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                
                # Process square format if selected
                if square_format:
                    output_filename = f"{uploaded_file.name.split('.')[0]}_square.mp4"
                    output_path = os.path.join(temp_dir, output_filename)
                    progress_text.text(f"Processing {uploaded_file.name} to square format...")
                    create_square_video(input_path, output_path)
                    current_conversion += 1
                    progress = current_conversion / total_conversions
                    progress_bar.progress(progress)
                    
                    # Add to processed videos list
                    st.session_state.processed_videos.append({
                        "name": output_filename,
                        "path": output_path,
                        "original_name": uploaded_file.name,
                        "format": "Square (1080x1080)"
                    })
                
                # Process square with blur format if selected
                if square_blur_format:
                    output_filename = f"{uploaded_file.name.split('.')[0]}_square_blur.mp4"
                    output_path = os.path.join(temp_dir, output_filename)
                    progress_text.text(f"Processing {uploaded_file.name} to square format with blur...")
                    create_square_blur_video(input_path, output_path)
                    current_conversion += 1
                    progress = current_conversion / total_conversions
                    progress_bar.progress(progress)
                    
                    # Add to processed videos list
                    st.session_state.processed_videos.append({
                        "name": output_filename,
                        "path": output_path,
                        "original_name": uploaded_file.name,
                        "format": "Square with Blur (1080x1080)"
                    })
                
                # Process landscape format if selected
                if landscape_format:
                    output_filename = f"{uploaded_file.name.split('.')[0]}_landscape.mp4"
                    output_path = os.path.join(temp_dir, output_filename)
                    progress_text.text(f"Processing {uploaded_file.name} to landscape format...")
                    create_landscape_video(input_path, output_path)
                    current_conversion += 1
                    progress = current_conversion / total_conversions
                    progress_bar.progress(progress)
                    
                    # Add to processed videos list
                    st.session_state.processed_videos.append({
                        "name": output_filename,
                        "path": output_path,
                        "original_name": uploaded_file.name,
                        "format": "Landscape (1920x1080)"
                    })
                
                # Clean up temp input file
                os.remove(input_path)
            
            if not st.session_state.stop_processing:
                progress_text.text("All videos processed!")
                st.success("✨ Videos have been converted successfully!")
        
        # Add stop button
        if col2.button("⏹️ Stop", type="secondary", use_container_width=True):
            st.session_state.stop_processing = True
    
    # Display preview and download section if there are processed videos
    if st.session_state.processed_videos:
        st.subheader("🎬 Preview and Download")
        st.write("Your converted videos are ready! Preview and download them below.")
        
        # Add a clear button to reset the processed videos
        if st.button("🗑️ Clear All Videos", type="secondary"):
            st.session_state.processed_videos = []
            st.experimental_rerun()
        
        # Add batch download option
        if len(st.session_state.processed_videos) > 1:
            st.info("💡 Tip: You can download individual videos or use the batch download option below.")
            
            # Create a zip file with all videos for batch download
            if st.button("📦 Download All Videos (ZIP)", type="primary"):
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
                            mime="application/zip"
                        )
        
        # Group videos by original file
        original_files = {}
        for video in st.session_state.processed_videos:
            if video["original_name"] not in original_files:
                original_files[video["original_name"]] = []
            original_files[video["original_name"]].append(video)
        
        # Create a container with a smaller width for the video display
        with st.container():
            # Use columns to create a more compact layout
            for original_name, videos in original_files.items():
                st.write(f"**Original file:** {original_name}")
                
                # Create a row for each video
                for video in videos:
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.write(f"**{video['format']}**")
                        
                        # Get video metadata
                        metadata = get_video_metadata(video["path"])
                        
                        # Display video preview with a smaller size
                        st.video(video["path"], start_time=0)
                        
                        # Display metadata
                        st.caption(f"⏱️ Duration: {metadata['duration']}")
                        st.caption(f"📊 Size: {metadata['size']}")
                    
                    with col2:
                        # Add download button in the second column
                        with open(video["path"], "rb") as file:
                            st.download_button(
                                label=f"📥 Download",
                                data=file,
                                file_name=video["name"],
                                mime="video/mp4",
                                key=f"download_{video['name']}",
                                use_container_width=True
                            )
                
                st.divider()
    
    # Add a footer with additional information
    st.markdown("---")
    st.markdown("### About")
    
    # Display example image
    st.image("exemple.png", caption="Format examples")
    
    st.markdown("""
    This app helps you convert your vertical videos to different formats suitable for various platforms:
    
    - 📱 **Square (1080x1080)**: Google Ads
    - 🎨 **Square with Blur (1080x1080)**: Google Ads ( keep the original 9x16 aspect ratio and add a blur effect)
    - 🖼️ **Landscape (1920x1080)**: YouTube ( keep the original 9x16 and add a blur effect)
    
    Made with ❤️ by Jiali
    """)

if __name__ == "__main__":
    main() 