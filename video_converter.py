import streamlit as st
import moviepy.editor as mp
from pathlib import Path
import numpy as np
from PIL import Image, ImageFilter
import os
import tempfile
import shutil
import time

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
mp.video.fx.resize.resize = patched_resize

def create_square_video(input_path, output_path):
    # Load the video
    video = mp.VideoFileClip(input_path)
    
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

def create_square_blur_video(input_path, output_path):
    # Load the video
    video = mp.VideoFileClip(input_path)
    
    # Force video to 30 FPS and calculate adjusted duration
    video = video.set_fps(30)
    frame_duration = 1.0/30  # Duration of one frame at 30fps
    exact_duration = video.duration - (4 * frame_duration)  # Remove 2 frames worth of duration
    
    # Target dimensions (square)
    target_size = 1080
    
    # Calculate scaling for portrait video in center
    scale = target_size/video.h  # Changed to always match height
    new_size = (int(video.w * scale), int(video.h * scale))
    
    # Resize original video for center
    center_video = video.resize(new_size)
    
    # Create blurred background - maintain aspect ratio while filling frame
    bg_scale = max(target_size/video.w, target_size/video.h)
    bg_size = (int(video.w * bg_scale), int(video.h * bg_scale))
    background = video.resize(bg_size)
    background = background.without_audio()  # This line ensures background has no audio
    
    # Calculate position to center the background
    bg_x = (target_size - bg_size[0]) // 2
    bg_y = (target_size - bg_size[1]) // 2
    background = background.set_position((bg_x, bg_y))
    
    # Apply stronger blur
    background = background.fl_image(lambda frame: np.array(
        Image.fromarray(frame)
        .filter(ImageFilter.GaussianBlur(radius=30))  # Increased blur radius
        .resize((bg_size[0], bg_size[1]))
    ))
    
    # Position center video
    x_center = (target_size - new_size[0]) // 2
    y_center = (target_size - new_size[1]) // 2
    center_video = center_video.set_position((x_center, y_center))
    
    # Composite final video
    final = mp.CompositeVideoClip([background, center_video], size=(target_size, target_size))
    
    # Improved audio handling with adjusted duration
    if center_video.audio is not None:
        # Ensure audio duration matches adjusted video duration
        audio = center_video.audio.subclip(0, exact_duration)
        final = final.set_audio(audio)
    
    # Set the duration of the final video to match the adjusted duration
    final = final.set_duration(exact_duration)
    
    # Write output with high quality settings
    try:
        final.write_videofile(
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
        final.close()
        if Path(output_path).exists():
            Path(output_path).unlink()
        raise e
    finally:
        # Clean up resources
        video.close()
        final.close()

def create_landscape_video(input_path, output_path):
    # Load the video
    video = mp.VideoFileClip(input_path)
    
    # Force video to 30 FPS and calculate adjusted duration
    video = video.set_fps(30)
    frame_duration = 1.0/30  # Duration of one frame at 30fps
    exact_duration = video.duration - (4 * frame_duration)  # Remove 2 frames worth of duration
    
    # Target dimensions
    target_width = 1920
    target_height = 1080
    
    # Calculate scaling for portrait video in center
    scale = target_height/video.h  # Changed to always match height
    new_size = (int(video.w * scale), int(video.h * scale))
    
    # Resize original video for center
    center_video = video.resize(new_size)
    
    # Create blurred background - maintain aspect ratio while filling frame
    bg_scale = max(target_width/video.w, target_height/video.h)
    bg_size = (int(video.w * bg_scale), int(video.h * bg_scale))
    background = video.resize(bg_size)
    background = background.without_audio()  # This line ensures background has no audio
    
    # Calculate position to center the background
    bg_x = (target_width - bg_size[0]) // 2
    bg_y = (target_height - bg_size[1]) // 2
    background = background.set_position((bg_x, bg_y))
    
    # Apply stronger blur
    background = background.fl_image(lambda frame: np.array(
        Image.fromarray(frame)
        .filter(ImageFilter.GaussianBlur(radius=30))  # Increased blur radius
        .resize((bg_size[0], bg_size[1]))
    ))
    
    # Position center video
    x_center = (target_width - new_size[0]) // 2
    center_video = center_video.set_position((x_center, 0))
    
    # Composite final video
    final = mp.CompositeVideoClip([background, center_video], size=(target_width, target_height))
    
    # Improved audio handling with adjusted duration
    if center_video.audio is not None:
        # Ensure audio duration matches adjusted video duration
        audio = center_video.audio.subclip(0, exact_duration)
        final = final.set_audio(audio)
    
    # Set the duration of the final video to match the adjusted duration
    final = final.set_duration(exact_duration)
    
    # Write output with high quality settings
    try:
        final.write_videofile(
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
        final.close()
        if Path(output_path).exists():
            Path(output_path).unlink()
        raise e
    finally:
        # Clean up resources
        video.close()
        final.close()

def get_video_metadata(video_path):
    """Get metadata for a video file"""
    try:
        video = mp.VideoFileClip(video_path)
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
                
                # Create a row for each video format
                for video in videos:
                    # Create a column for each video with a smaller width
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
    st.markdown("""
    This app helps you convert your vertical videos to different formats suitable for various platforms:
    
    - 📱 **Square (1080x1080)**: Google Ads
    - 🎨 **Square with Blur (1080x1080)**: Google Ads ( keep the original 9x16 aspect ratio and add a blur effect)
    - 🖼️ **Landscape (1920x1080)**: YouTube ( keep the original 9x16 and add a blur effect)
    
    Made with ❤️ by Jiali
    """)

if __name__ == "__main__":
    main() 