from dataclasses import dataclass, field
from typing import Optional


class StaticConfig(object):
    # Max initial bitrate (bps)
    max_initial_bitrate = 1000000

    # averageSpeed = SMOOTHING_FACTOR * lastSpeed + (1-SMOOTHING_FACTOR) * averageSpeed;
    smoothing_factor = 0.5

    # minimum frame chunk size ratio
    # The size ratio of a segment which is for I-, P-, and B-frames.
    min_frame_chunk_ratio = 0.6

    # VQ threshold
    vq_threshold = 0.8

    # [Not Used] VQ threshold for size ratio
    vq_threshold_size_ratio = min_frame_chunk_ratio * (min_frame_chunk_ratio + (1 - min_frame_chunk_ratio) * vq_threshold)

    # Update interval
    update_interval = 0.05

    # [Not Used] Chunk size
    chunk_size = 40960

    # [Not Used] Timeout max ratio
    timeout_max_ratio = 2

    # [Not Used] Min Duration for quality increase (ms)
    min_duration_for_quality_increase_ms = 6000

    # [Not Used] Max duration for quality decrease (ms)
    max_duration_for_quality_decrease_ms = 8000

    # [Not Used] Min duration to retrain after discard (ms)
    min_duration_to_retrain_after_discard_ms = 8000

    # [Not Used] Bandwidth fraction
    bandwidth_fraction = 0.75

    # If the packet arrives later than this it should not be consider in bw estimation
    max_packet_delay = 2

    # Continuous bw estimation window (s)
    cont_bw_window = 1


@dataclass
class PlayerConfig:
    static = StaticConfig

    # Required config
    input: str = ""
    trace: str = ""
    run_dir: str = ""
    log: str = ""

    time_factor: float = 1

    # Modules
    mod_mpd: str = "mpd"
    mod_downloader: str = "auto"
    mod_bw: str = "bw_meter"
    mod_nes: str = "bones"
    mod_scheduler: str = "scheduler"
    mod_download_buffer: str = "download_buffer"
    mod_enhance_buffer: str = "enhance_buffer"
    mod_player: str = "dash"
    mod_enhancer: str = "imdn"
    mod_renderer: str = "headless"
    # mod_renderer: str = "opengl"
    mod_analyzer: list[str] = field(default_factory=lambda: ["data_collector"])

    # Buffer Configuration
    buffer_duration: float = 60  # maximum buffer level (s)
    safe_buffer_level: float = 6
    panic_buffer_level: float = 2.5

    # min_rebuffer_duration: float = 1
    # min_start_duration: float = 1

    select_as: str = "-"

    ssl_keylog_file: Optional[str] = None

    # Live event logs file path
    live_log: Optional[str] = None

    # Display configuration
    # display_width = 1920
    # display_height = 1080
    display_width = 1280
    display_height = 720
    # display_width = 640
    # display_height = 360
    display_fps = 30  # target display fps, cannot guarantee
    # display_fps = 60

    # Device configuration
    enhancer_device = "cpu"
    # enhancer_device = "cuda"
    # renderer_device = "cpu"
    renderer_device = "cuda"

    # Enhancement metadata
    content_aware = False
    if content_aware:
        enhance_metadata = "imdn_bbb_quality.json"
    else:
        enhance_metadata = "imdn_div2k_quality.json"

    def validate(self) -> None:
        """Assert if config properties are set properly"""
        assert bool(self.input), "A non-empty '--input' arg or 'input' config is required"

