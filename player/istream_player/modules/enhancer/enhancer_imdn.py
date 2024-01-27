import asyncio
import json
import logging

from istream_player.config.config import PlayerConfig
# from istream_player.core.buffer import BufferManager
from istream_player.core.module import Module, ModuleOption
from istream_player.core.mpd_provider import MPDProvider
from istream_player.core.enhancer import Enhancer, EnhancerEventListener
from istream_player.core.scheduler import Scheduler
from istream_player.models import State
from istream_player.utils.async_utils import critical_task

from istream_player.modules.download_buffer import DownloadBufferImpl
from istream_player.modules.enhance_buffer import EnhanceBufferImpl
from istream_player.modules.decoder import DecoderNvCodec, TensorConverter
from istream_player.core.player import PlayerEventListener
from istream_player.models.mpd_objects import Segment
from istream_player.core.player import Player
from istream_player.core.downloader import (DownloadManager, DownloadRequest, DownloadType)
from .imdn_model import IMDN, IMDN_RTC
import torch
import os
import time
import PyNvCodec as nvc
import numpy as np
from typing import Dict, List
import tempfile


@ModuleOption("imdn", default=True, requires=["model_downloader", DownloadBufferImpl, EnhanceBufferImpl, Scheduler, Player])
class IMDNEnhancer(Module, Enhancer, PlayerEventListener):
    log = logging.getLogger("IMDNEnhancer")

    def __init__(self):
        super().__init__()
        self._accessible = asyncio.Condition()
        self._is_ready = False

        self.device = None
        self.warmup_epoch = 3
        self.measure_epoch = 5

        self.config = None
        self.display_W = None
        self.display_H = None
        self.enhance_buffer = None
        self.scheduler = None
        self.resolution_set = None
        self.latency_table = np.zeros((5, 5))
        self.quality_table = None

        self.model_pool = None
        self.tensor_converter = None
        self.seg_time = 4
        self.frame_rate = 30
        self.time_factor = 1.  # enhancement speed variation factor
        self.task_start = None
        self.task_total = None
        self.old_quality = None

        self.check_abort = False
        self.played_urls = []


    async def setup(self,
                    config: PlayerConfig,
                    model_downloader: DownloadManager,
                    download_buffer: DownloadBufferImpl,
                    enhance_buffer: EnhanceBufferImpl,
                    scheduler: Scheduler,
                    player: Player,
                    **kwargs
    ):
        self.config = config
        self.display_W = config.display_width
        self.display_H = config.display_height
        self.device = config.enhancer_device
        self.content_aware = config.content_aware
        self.run_dir = config.run_dir

        self.download_buffer = download_buffer
        self.enhance_buffer = enhance_buffer
        self.scheduler = scheduler
        self.download_manager = model_downloader

        player.add_listener(self)

        self.quality_table = self.get_quality_table()
        # self.log.info("Quality table: {}".format(self.quality_table))

    async def start(self, adaptation_sets):
        async with self._accessible:
            self.model_pool = await self.load_model()

            self.resolution_set = self.get_resolution_set(adaptation_sets)
            # self.log.info("Resolution set: {}".format(self.resolution_set))

            self.latency_table = await self.measure_latency(self.resolution_set, self.model_pool)
            # self.log.info("Latency table: {}".format(self.latency_table))

            self.frame_rate = self.get_frame_rate(adaptation_sets)
            self.seg_time = self.scheduler.mpd_provider.mpd.max_segment_duration

            if self.old_quality is not None:
                self.quality_table = self.old_quality
                # self.log.info("Quality table: {}".format(self.quality_table))

            self._accessible.notify_all()
            self._is_ready = True
        return

    def get_frame_rate(self, adaptation_sets):
        for as_idx in adaptation_sets:
            as_obj = adaptation_sets[as_idx]
            if as_obj.content_type != "video":
                continue
            frame_rate = np.array(as_obj.frame_rate).astype(float)
            return frame_rate

    def is_ready(self):
        return self._is_ready

    def get_resolution_set(self, adaptation_sets):
        resolution_set = {}
        for as_idx in adaptation_sets:
            as_obj = adaptation_sets[as_idx]
            if as_obj.content_type != "video":
                continue
            for repr_idx in as_obj.representations:
                repr_obj = as_obj.representations[repr_idx]
                scale = min(int(self.display_W / repr_obj.width), int(self.display_H / repr_obj.height))
                # only enhance low-resolution videos
                if scale != 1:
                    resolution_set[repr_idx] = ((repr_obj.width, repr_obj.height, scale))
            break  # assume only one video track
        return resolution_set

    @critical_task()
    async def run(self):
        async with self._accessible:
            await self._accessible.wait()

        while self._is_ready:
            # Enhance segment
            index, segments = await self.enhance_buffer.dequeue()

            if self.enhance_buffer.is_empty():
                if self.scheduler.is_end:
                    self._is_ready = False
                    self.log.info("Enhancer closed")
                    return

            for listener in self.listeners:
                await listener.on_enhancement_start(segments)

            for as_idx in segments:
                segment = segments[as_idx]
                level = segment.enhance_action
                abort = False

                self.log.info(f"Enhancing segment index: {index}, download action: {segment.download_action}, enhance action: {level}")

                start_time = time.time()
                self.task_start = start_time
                self.task_total = self.get_latency_table()[segment.download_action, segment.enhance_action]

                # no enhancement
                if level == 0:
                    abort = True
                repr_idx = segment.repr_id
                # maximum resolution
                if repr_idx not in self.resolution_set:
                    abort = True
                # played segment
                if segment.url in self.played_urls:
                    abort = True
                if abort:
                    self.log.info(f"Abort enhancing segment index: {index}, download action: {segment.download_action}, enhance action: {segment.enhance_action}")
                    break

                _, _, scale = self.resolution_set[repr_idx]

                decoder = DecoderNvCodec(self.config, segment, resize=False)
                decode_W, decode_H = decoder.resolution()
                display_W, display_H = self.config.display_width, self.config.display_height
                tensor_converter = TensorConverter(decode_W, decode_H, display_W, display_H, gpu_id=0)
                model = self.model_pool[(scale, level)]
                result : List[torch.Tensor] = []
                cnt = 0

                while True:
                    surf = decoder.decode_one_frame()
                    await asyncio.sleep(0)

                    if surf is None:
                        break

                    surf_enh = await self.enhance_one_frame(surf, model, tensor_converter)
                    result.append(surf_enh)
                    cnt += 1

                    await asyncio.sleep(0)  # yield to other tasks

                    if self.check_abort and segment.url in self.played_urls:
                        abort = True
                        self.check_abort = False
                        break
                    else:
                        self.check_abort = False

                if abort:
                    self.log.info(f"Abort enhancing segment index: {index}, download action: {segment.download_action}, enhance action: {segment.enhance_action}")
                    break

                segment.decode_data = result

                # Replace the original segment with the enhanced one
                await self.download_buffer.replace(index, {as_idx: segment})

                end_time = time.time()

                self.latency_table[segment.download_action, segment.enhance_action] = (end_time - start_time) / (self.seg_time * self.frame_rate)
                self.log.info(f"Complete enhancing segment index: {index}, download action: {segment.download_action}, enhance action: {segment.enhance_action}, latency: {end_time - start_time:.3f}, time factor: {self.time_factor:.3f}")

    async def enhance_one_frame(self, surf: nvc.Surface, model: torch.nn.Module, tensor_converter: TensorConverter) -> torch.Tensor:
        tensor = tensor_converter.surface_to_tensor(surf)
        tensor = tensor.to(self.device)
        await asyncio.sleep(0)
        with torch.no_grad():
            tensor = model(tensor)
            await asyncio.sleep(0)
            tensor = torch.nn.functional.interpolate(tensor, size=(self.display_H, self.display_W), mode='bicubic', align_corners=False)
            await asyncio.sleep(0)
        tensor = tensor.cpu()  # save on CPU to save GPU memory
        await asyncio.sleep(0)
        return tensor

    async def load_model(self):
        file_name = "imdn_path.json"
        model_path = json.load(open(file_name))
        if self.content_aware:
            model_path = model_path["aware"]
        else:
            model_path = model_path["agnostic"]

        model_pool = {}
        ABSOLUTE_PATH = os.path.dirname(os.path.abspath(__file__))
        for scale in [2, 3, 4]:
            for level in [1, 2, 3]:
                url = os.path.join(ABSOLUTE_PATH, model_path[f"scale{scale}_level{level}"])

                if level == 1:
                    model = IMDN_RTC(upscale=scale, num_modules=3, nf=6)  # low
                    model = await self._download_and_load(model, url, download=self.content_aware)
                elif level == 2:
                    model = IMDN_RTC(upscale=scale)  # medium
                    model = await self._download_and_load(model, url, download=self.content_aware)
                elif level == 3 and (not self.content_aware):
                    model = IMDN(upscale=scale, nf=32)  # high
                    model = await self._download_and_load(model, url, download=self.content_aware)


                model_pool[(scale, level)] = model.to(self.device)

        return model_pool

    async def _download_and_load(self, model, url, download=False):
        if download:
            await self.download_manager.download(DownloadRequest(url, DownloadType.STREAM_INIT))
            model_data, _ = await self.download_manager.wait_complete(url)
            model_file = tempfile.NamedTemporaryFile(dir=self.run_dir, delete=False)
            model_file.write(model_data)
            model_file.close()
            url = model_file.name
            pass

        model.load_state_dict(torch.load(url))
        return model


    async def measure_latency(self, resolution_set, model_pool, file_name="imdn_latency.json"):
        """
        Warm up enhancement models and measure their latency
        """
        if os.path.exists(file_name):
            latency_table = json.load(open(file_name))
            return np.array(latency_table)

        self.log.info("Start measuring latency (only for the first time)")
        latency_set = {}
        for repr_idx in resolution_set:
            width, height, scale = resolution_set[repr_idx]
            for level in [1, 2, 3]:
                data_pool = []
                for i in range(self.warmup_epoch + self.measure_epoch):
                    data_pool.append(torch.rand((1, 3, height, width)).to(self.device))
                    await asyncio.sleep(0)

                model = model_pool[(scale, level)]
                model.eval()
                with torch.no_grad():
                    for i in range(self.warmup_epoch):
                        tensor = data_pool[i]
                        model(tensor)
                        await asyncio.sleep(0)

                    torch.cuda.synchronize()
                    start_time = time.time()
                    for i in range(self.measure_epoch):
                        tensor = data_pool[i + self.warmup_epoch]
                        model(tensor)
                        await asyncio.sleep(0)

                    torch.cuda.synchronize()
                    end_time = time.time()
                    latency_set[(scale, level)] = (end_time - start_time) / self.measure_epoch

        # bitrate (240p, 360p, 480p, 720p, 1080p), level (no, low, medium, high, ultra)
        latency_table = np.zeros((5, 5))
        for setting in latency_set:
            repr_idx, level = setting
            latency_table[4 - repr_idx, level] = latency_set[(repr_idx, level)]
        json.dump(latency_table.tolist(), open(file_name, "w"), indent=4)
        return latency_table

    def get_latency_table(self):
        try:
            return self.latency_table * self.time_factor * self.seg_time * self.frame_rate
        except:
            return None

    # assume enhancement quality metadata exist locally
    def get_quality_table(self):
        if self.content_aware:
            file_name = "imdn_bbb_quality.json"
        else:
            file_name = "imdn_div2k_quality.json"

        if self.quality_table is not None:
            return self.quality_table

        quality_table = json.load(open(file_name))
        quality_table = np.array(quality_table)
        quality_table[quality_table < 0] = -np.inf  # invalid enhancement

        if self.content_aware:
            self.old_quality = quality_table.copy()
            self.old_quality[:, -2:] = -np.inf  # only low and medium level
            quality_table[:, 1:] = -np.inf

        self.quality_table = quality_table
        return quality_table

    def remain_task(self):
        if self.task_start is None:
            return 0
        return max(self.task_total - (time.time() - self.task_start), 0)

    async def on_segment_playback_start(self, segments: Dict[int, Segment]):
        # abort ongoing and future tasks
        for idx in segments:
            segment = segments[idx]
            self.played_urls.append(segment.url)
        self.check_abort = True
        return

    async def cleanup(self) -> None:
        self._is_ready = False
        await self.enhance_buffer.cleanup()
        return