from pathlib import Path
from typing import List, Dict, Any, Optional
import cv2
import whisper
import torch
import numpy as np
from moviepy.editor import VideoFileClip
import tempfile
import subprocess
from loguru import logger

from .base import BaseContentProcessor, ProcessedContent, ContentMetadata


class VideoProcessor(BaseContentProcessor):
    def __init__(self, whisper_model: str = "base", device: str = "cuda"):
        super().__init__()
        self.supported_formats = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
        self.device = device if torch.cuda.is_available() else "cpu"
        
        logger.info(f"Loading Whisper model: {whisper_model} on {self.device}")
        self.whisper_model = whisper.load_model(whisper_model, device=self.device)
    
    def validate_file(self, file_path: Path) -> bool:
        return file_path.exists() and file_path.suffix.lower() in self.supported_formats
    
    async def process(self, file_path: Path) -> ProcessedContent:
        if not self.validate_file(file_path):
            raise ValueError(f"Invalid video file: {file_path}")
        
        logger.info(f"Processing video: {file_path}")
        
        video_info = self._extract_video_info(file_path)
        audio_transcript = self._extract_audio_transcript(file_path)
        keyframes = self._extract_keyframes(file_path)
        scene_analysis = self._analyze_scenes(file_path)
        
        structured_content = {
            'video_info': video_info,
            'transcript': audio_transcript,
            'keyframes': keyframes,
            'scenes': scene_analysis
        }
        
        full_text = self._create_text_representation(audio_transcript, scene_analysis)
        
        metadata = self._extract_video_metadata(file_path, video_info)
        
        return ProcessedContent(
            raw_text=full_text,
            structured_content=[structured_content],
            metadata=metadata
        )
    
    def _extract_video_info(self, file_path: Path) -> Dict[str, Any]:
        try:
            video = cv2.VideoCapture(str(file_path))
            
            info = {
                'width': int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'fps': video.get(cv2.CAP_PROP_FPS),
                'frame_count': int(video.get(cv2.CAP_PROP_FRAME_COUNT)),
                'duration': video.get(cv2.CAP_PROP_FRAME_COUNT) / video.get(cv2.CAP_PROP_FPS)
            }
            
            video.release()
            return info
        except Exception as e:
            logger.error(f"Failed to extract video info: {e}")
            return {}
    
    def _extract_audio_transcript(self, file_path: Path) -> Dict[str, Any]:
        try:
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_audio:
                temp_audio_path = temp_audio.name
            
            video = VideoFileClip(str(file_path))
            if video.audio:
                video.audio.write_audiofile(temp_audio_path, verbose=False, logger=None)
            else:
                logger.warning(f"No audio track found in {file_path}")
                return {'segments': [], 'text': '', 'language': None}
            
            logger.info("Transcribing audio with Whisper...")
            result = self.whisper_model.transcribe(
                temp_audio_path,
                language=None,
                task='transcribe',
                verbose=False
            )
            
            Path(temp_audio_path).unlink(missing_ok=True)
            
            return {
                'text': result.get('text', ''),
                'language': result.get('language', 'unknown'),
                'segments': [
                    {
                        'start': seg['start'],
                        'end': seg['end'],
                        'text': seg['text'].strip()
                    }
                    for seg in result.get('segments', [])
                ]
            }
        except Exception as e:
            logger.error(f"Audio transcription failed: {e}")
            return {'segments': [], 'text': '', 'language': None}
    
    def _extract_keyframes(self, file_path: Path, num_frames: int = 10) -> List[Dict[str, Any]]:
        keyframes = []
        
        try:
            video = cv2.VideoCapture(str(file_path))
            total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = video.get(cv2.CAP_PROP_FPS)
            
            if total_frames == 0:
                return keyframes
            
            frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
            
            for idx in frame_indices:
                video.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = video.read()
                
                if ret:
                    timestamp = idx / fps
                    
                    keyframes.append({
                        'frame_number': int(idx),
                        'timestamp': float(timestamp),
                        'frame_shape': frame.shape,
                        'mean_intensity': float(np.mean(frame))
                    })
            
            video.release()
        except Exception as e:
            logger.error(f"Keyframe extraction failed: {e}")
        
        return keyframes
    
    def _analyze_scenes(self, file_path: Path) -> List[Dict[str, Any]]:
        scenes = []
        
        try:
            video = cv2.VideoCapture(str(file_path))
            fps = video.get(cv2.CAP_PROP_FPS)
            prev_frame = None
            scene_start = 0
            frame_count = 0
            threshold = 30.0
            
            while True:
                ret, frame = video.read()
                if not ret:
                    break
                
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                if prev_frame is not None:
                    diff = cv2.absdiff(prev_frame, gray)
                    mean_diff = np.mean(diff)
                    
                    if mean_diff > threshold:
                        scene_end = frame_count / fps
                        scenes.append({
                            'scene_number': len(scenes) + 1,
                            'start_time': float(scene_start),
                            'end_time': float(scene_end),
                            'duration': float(scene_end - scene_start)
                        })
                        scene_start = scene_end
                
                prev_frame = gray
                frame_count += 1
            
            if frame_count > 0:
                scenes.append({
                    'scene_number': len(scenes) + 1,
                    'start_time': float(scene_start),
                    'end_time': float(frame_count / fps),
                    'duration': float((frame_count / fps) - scene_start)
                })
            
            video.release()
        except Exception as e:
            logger.error(f"Scene analysis failed: {e}")
        
        return scenes
    
    def _create_text_representation(
        self, 
        transcript: Dict[str, Any],
        scenes: List[Dict[str, Any]]
    ) -> str:
        text = "VIDEO CONTENT ANALYSIS\n\n"
        
        if transcript.get('text'):
            text += "TRANSCRIPT:\n"
            text += transcript['text'] + "\n\n"
        
        if transcript.get('segments'):
            text += "TIMESTAMPED SEGMENTS:\n"
            for seg in transcript['segments']:
                text += f"[{seg['start']:.2f}s - {seg['end']:.2f}s]: {seg['text']}\n"
            text += "\n"
        
        if scenes:
            text += f"SCENE BREAKDOWN ({len(scenes)} scenes detected):\n"
            for scene in scenes:
                text += f"Scene {scene['scene_number']}: {scene['start_time']:.2f}s - {scene['end_time']:.2f}s ({scene['duration']:.2f}s)\n"
        
        return text
    
    def _extract_video_metadata(self, file_path: Path, video_info: Dict[str, Any]) -> ContentMetadata:
        base_metadata = self.extract_metadata(file_path)
        
        base_metadata.duration = video_info.get('duration', 0)
        base_metadata.metadata = {
            'width': video_info.get('width'),
            'height': video_info.get('height'),
            'fps': video_info.get('fps'),
            'frame_count': video_info.get('frame_count')
        }
        
        return base_metadata
