# liquid_time_node.py

import os
import torch
import numpy as np
import urllib.request
from tqdm import tqdm
import math

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

class LiquidTimeNode:
    def __init__(self):
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.precision = torch.float16 if torch.cuda.is_available() else torch.float32
        
        print(f"[LiquidTime] Initializing with device: {self.device}, precision: {self.precision}")
        
        self.model_urls = {
            'fp16': 'https://github.com/dajes/frame-interpolation-pytorch/releases/download/v1.0.2/film_net_fp16.pt',
            'fp32': 'https://github.com/dajes/frame-interpolation-pytorch/releases/download/v1.0.2/film_net_fp32.pt'
        }
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "target_frames": ("INT", {"default": 60, "min": 2, "max": 1000}),
                "start_curve": ("INT", {"default": 0, "min": -100, "max": 100}),
                "middle_curve": ("INT", {"default": 0, "min": -100, "max": 100}),
                "middle_moment": ("INT", {"default": 50, "min": 0, "max": 100}),
                "end_curve": ("INT", {"default": 0, "min": -100, "max": 100})
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "interpolate_frames"
    CATEGORY = "animation/LiquidTime"

    def download_model(self, model_path):
        """Download model with progress bar"""
        model_type = 'fp16' if self.precision == torch.float16 else 'fp32'
        url = self.model_urls[model_type]
        
        print(f"\n[LiquidTime] Creating model directory...")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        print(f"[LiquidTime] Starting download of {model_type} model from {url}")
        try:
            with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, 
                                   desc=f"[LiquidTime] Downloading {model_type} model",
                                   bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as t:
                urllib.request.urlretrieve(
                    url,
                    filename=model_path,
                    reporthook=t.update_to
                )
            print(f"[LiquidTime] Successfully downloaded {model_type} model to {model_path}")
        except Exception as e:
            print(f"[LiquidTime] Error downloading model: {str(e)}")
            if os.path.exists(model_path):
                os.remove(model_path)
            raise

    def get_model_path(self):
        """Get path to model and ensure it exists"""
        base_path = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(base_path, 'models')
        os.makedirs(models_dir, exist_ok=True)
        
        model_type = 'fp16' if self.precision == torch.float16 else 'fp32'
        model_path = os.path.join(models_dir, f'film_net_{model_type}.pt')
        
        if not os.path.exists(model_path):
            print(f"[LiquidTime] Model not found at {model_path}")
            self.download_model(model_path)
            
        return model_path

    def load_model(self):
        if self.model is None:
            print("[LiquidTime] Loading model...")
            model_path = self.get_model_path()
            try:
                self.model = torch.jit.load(model_path, map_location='cpu')
                self.model.eval().to(device=self.device, dtype=self.precision)
                print(f"[LiquidTime] Successfully loaded model with precision {self.precision} on {self.device}")
            except Exception as e:
                print(f"[LiquidTime] Error loading model: {str(e)}")
                raise RuntimeError(f"Error loading model: {str(e)}")

    def sample_frames(self, source_frames, target_count):
        """
        Wybiera target_count klatek z source_frames w równych odstępach czasowych.
        """
        if target_count >= len(source_frames):
            return source_frames
            
        print(f"[LiquidTime] Reducing frames from {len(source_frames)} to {target_count}")
        
        # Obliczamy indeksy klatek do zachowania
        indices = np.linspace(0, len(source_frames) - 1, target_count)
        indices = np.round(indices).astype(int)
        
        # Wybieramy klatki
        selected_frames = [source_frames[i] for i in indices]
        
        print(f"[LiquidTime] Selected frame indices: {indices}")
        return selected_frames

    def create_time_mapping(self, num_points, start_curve, middle_curve, middle_moment, end_curve):
        """
        Ulepszone mapowanie czasu z mocniejszym wpływem parametrów.
        """
        print(f"[LiquidTime] Creating enhanced time mapping with curves:")
        print(f"  - Start curve: {start_curve}")
        print(f"  - Middle curve: {middle_curve} at {middle_moment}%")
        print(f"  - End curve: {end_curve}")
        
        # Konwertujemy middle_moment z procentów na zakres 0-1
        middle_t = middle_moment / 100.0
        
        # Generujemy bazowe punkty czasowe
        t_values = np.linspace(0, 1, num_points)
        
        # Wzmocnienie efektu krzywych
        curve_power = 2.5  # Zwiększa intensywność efektu
        
        def enhanced_curve_influence(value):
            """Wzmacnia wpływ krzywej zachowując znak"""
            return np.sign(value) * abs(value / 100.0) ** 0.75 * curve_power
        
        # Konwertujemy wartości krzywych
        start_influence = enhanced_curve_influence(start_curve)
        middle_influence = enhanced_curve_influence(middle_curve)
        end_influence = enhanced_curve_influence(end_curve)
        
        def transform_time(t):
            # Bazowa transformacja dla middle_moment (aktywna nawet bez middle_curve)
            middle_shift = (middle_t - 0.5) * 0.5  # Maksymalne przesunięcie o 25% w każdą stronę
            t_shifted = t + middle_shift * np.sin(np.pi * t)  # Sinusoidalne przesunięcie
            
            # Obliczamy wpływy poszczególnych krzywych
            start_factor = np.exp(-4 * t_shifted) * start_influence
            end_factor = np.exp(-4 * (1 - t_shifted)) * end_influence
            
            # Wpływ middle_curve z gaussowskim rozkładem wokół middle_moment
            middle_width = 0.3  # Szerszy zakres wpływu
            middle_gaussian = np.exp(-((t_shifted - middle_t) ** 2) / (2 * middle_width ** 2))
            middle_factor = middle_gaussian * middle_influence
            
            # Łączymy wszystkie wpływy
            combined_shift = (start_factor + middle_factor + end_factor)
            
            # Aplikujemy przesunięcie z zachowaniem granic
            transformed_t = t_shifted + combined_shift
            
            # Zapewniamy monotoniczność (zawsze do przodu)
            transformed_t = t + (transformed_t - t) * 0.7  # 0.7 to siła efektu
            
            return transformed_t
        
        # Aplikujemy transformację do wszystkich punktów
        transformed_points = np.array([transform_time(t) for t in t_values])
        
        # Normalizujemy do zakresu [0, 1] zachowując proporcje
        transformed_points = (transformed_points - transformed_points.min()) / \
                           (transformed_points.max() - transformed_points.min())
        
        # Wygładzamy przejścia z zachowaniem kształtu
        def smooth_points(points, window=5):
            smoothed = np.copy(points)
            half_window = window // 2
            
            for i in range(half_window, len(points) - half_window):
                # Ważone średnia z większym wpływem punktów centralnych
                weights = np.exp(-0.5 * (np.arange(window) - half_window) ** 2)
                window_slice = points[i - half_window:i + half_window + 1]
                smoothed[i] = np.sum(window_slice * weights) / np.sum(weights)
            
            # Zachowujemy początek i koniec
            smoothed[:half_window] = points[:half_window]
            smoothed[-half_window:] = points[-half_window:]
            return smoothed
        
        smoothed_points = smooth_points(transformed_points)
        
        # Końcowa normalizacja
        smoothed_points = (smoothed_points - smoothed_points[0]) / \
                         (smoothed_points[-1] - smoothed_points[0])
        
        # Debug info
        print(f"[LiquidTime] Time mapping statistics:")
        print(f"  - Average speed: {1/(len(smoothed_points)-1):.4f}")
        print(f"  - Max speed: {np.max(np.diff(smoothed_points)):.4f}")
        print(f"  - Min speed: {np.min(np.diff(smoothed_points)):.4f}")
        print(f"  - Middle point actual position: {np.argmin(np.abs(smoothed_points - 0.5))/len(smoothed_points)*100:.1f}%")
        
        return smoothed_points

    def interpolate_pair(self, img1, img2, time_step):
        """Interpolate between two frames"""
        # Convert images to torch tensors in the correct format
        img1_tensor = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0)
        img2_tensor = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0)
        
        # Move to device and cast to correct precision
        img1_tensor = img1_tensor.to(device=self.device, dtype=self.precision)
        img2_tensor = img2_tensor.to(device=self.device, dtype=self.precision)
        
        dt = img1_tensor.new_full((1, 1), float(time_step))
        
        with torch.no_grad():
            try:
                interpolated = self.model(img1_tensor, img2_tensor, dt)
                return interpolated.squeeze(0).permute(1, 2, 0).cpu().numpy()
            except RuntimeError as e:
                if "out of memory" in str(e):
                    torch.cuda.empty_cache()
                    print("[LiquidTime] GPU out of memory error occurred!")
                    raise RuntimeError("GPU out of memory. Try reducing image size or using CPU mode.")
                raise

    def interpolate_frames(self, images, target_frames, start_curve=0, middle_curve=0, middle_moment=50, end_curve=0):
        print(f"\n[LiquidTime] Starting interpolation process...")
        print(f"[LiquidTime] Input: {len(images)} frames, Target: {target_frames} frames")
        print(f"[LiquidTime] Curve parameters: start={start_curve}, middle={middle_curve} at {middle_moment}%, end={end_curve}")
        
        # Konwertujemy input na numpy jeśli potrzeba
        if torch.is_tensor(images):
            images = images.cpu().numpy()
        
        num_source_frames = len(images)
        
        # Obsługa redukcji klatek
        if target_frames < num_source_frames:
            print(f"[LiquidTime] Reducing frames from {num_source_frames} to {target_frames}")
            images = self.sample_frames(images, target_frames)
            return (torch.from_numpy(np.stack(images)),)
        
        # Obsługa równej liczby klatek
        if target_frames == num_source_frames:
            print(f"[LiquidTime] Target frames equals source frames, returning without changes")
            return (torch.from_numpy(np.stack(images)),)
        
        # Interpolacja
        self.load_model()
        
        # Obliczamy punkty czasowe używając krzywej czasowej
        time_points = self.create_time_mapping(target_frames, start_curve, middle_curve, middle_moment, end_curve)
        
        # Obliczamy ile klatek przypada na każdy segment źródłowy
        source_segments = np.linspace(0, 1, num_source_frames)
        output_frames = []
        
        try:
            # Main interpolation loop with progress bar
            segment_progress = 0
            last_progress_update = 0
            progress_update_interval = max(1, target_frames // 100)  # Aktualizacja co 1% postępu
            
            with tqdm(total=target_frames, 
                     desc="[LiquidTime] Generating frames",
                     bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:
                
                current_source_idx = 0
                for frame_idx, t in enumerate(time_points):
                    # Znajdujemy odpowiedni segment źródłowy
                    while (current_source_idx < len(source_segments) - 1 and 
                           t > source_segments[current_source_idx + 1]):
                        current_source_idx += 1
                        segment_progress = 0
                    
                    if current_source_idx >= len(images) - 1:
                        current_source_idx = len(images) - 2
                    
                    # Obliczamy lokalny time_step w aktualnym segmencie
                    local_segment_start = source_segments[current_source_idx]
                    local_segment_end = source_segments[current_source_idx + 1]
                    local_t = (t - local_segment_start) / (local_segment_end - local_segment_start)
                    local_t = max(0, min(1, local_t))  # Clamp to [0, 1]
                    
                    # Obliczamy procent ukończenia aktualnego segmentu
                    segment_progress = local_t * 100
                    
                    # Aktualizujemy informacje o postępie
                    if frame_idx - last_progress_update >= progress_update_interval:
                        overall_progress = (frame_idx / target_frames) * 100
                        print(f"[LiquidTime] Processing: Segment {current_source_idx + 1}/{len(images) - 1} "
                              f"({segment_progress:.1f}% complete) | "
                              f"Overall: {overall_progress:.1f}% | "
                              f"Frames: {frame_idx}/{target_frames}")
                        last_progress_update = frame_idx
                    
                    # Interpolujemy klatkę
                    interpolated = self.interpolate_pair(
                        images[current_source_idx],
                        images[current_source_idx + 1],
                        local_t
                    )
                    output_frames.append(interpolated)
                    pbar.update(1)

                    # Informacja o rozpoczęciu nowego segmentu
                    if segment_progress == 0:
                        print(f"[LiquidTime] Starting new segment: {current_source_idx + 1}/{len(images) - 1}")
                    
                    # Okresowe czyszczenie pamięci GPU
                    if frame_idx % 100 == 0:
                        torch.cuda.empty_cache()
            
            final_frame_count = len(output_frames)
            print(f"[LiquidTime] Generated frames breakdown:")
            print(f"  - Initial source frames: {num_source_frames}")
            print(f"  - Total output frames: {final_frame_count}")
            print(f"  - Target frames: {target_frames}")
            
            if final_frame_count != target_frames:
                print(f"[LiquidTime] WARNING: Frame count mismatch!")
                print(f"  Expected: {target_frames}, Got: {final_frame_count}")
            else:
                print(f"[LiquidTime] Frame count matches target exactly!")
            
            # Convert back to torch tensor for ComfyUI compatibility
            result = torch.from_numpy(np.stack(output_frames))
            print(f"[LiquidTime] Output shape: {result.shape}")
            print(f"[LiquidTime] Process completed successfully!")
            
            # Dodajemy informacje o autorze i wsparciu
            print("\n" + "="*50)
            print("LiquidTime - by PabloGFX")
            print("YouTube: https://youtube.com/@lazniak")
            print("For support: https://buymeacoffee.com/eyb8tkx3to")
            print("="*50 + "\n")
            
            return (result,)
            
        except Exception as e:
            print(f"[LiquidTime] Error during interpolation: {str(e)}")
            raise

NODE_CLASS_MAPPINGS = {
    "LiquidTime": LiquidTimeNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LiquidTime": "LiquidTime - by PabloGFX"
}