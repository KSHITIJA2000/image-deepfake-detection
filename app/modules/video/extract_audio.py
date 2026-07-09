from moviepy import VideoFileClip
import os


def extract_audio_from_video(video_path, output_audio_path):

    video = None

    try:
        video = VideoFileClip(video_path)

        # -----------------------------
        # SAFE CHECK (NO AUDIO CASE)
        # -----------------------------
        if video.audio is None:
            print("[INFO] No audio track found in video.")
            return None

        # -----------------------------
        # EXTRACT AUDIO
        # -----------------------------
        video.audio.write_audiofile(
            output_audio_path,
            logger=None
        )

        return output_audio_path

    except Exception as e:
        print(f"Audio extraction error (safe skip): {e}")
        return None

    finally:
        # ALWAYS RELEASE RESOURCES
        try:
            if video:
                video.close()
        except:
            pass