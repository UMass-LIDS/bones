import asyncio
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Tuple

from istream_player.config.config import PlayerConfig
from istream_player.core.downloader import (DownloadEventListener,
                                            DownloadManager, DownloadRequest)
from istream_player.core.module import Module, ModuleOption

import time
import json
import numpy as np
from dataclasses import dataclass
import logging

@dataclass
class Trace:
    duration: np.array(float)  # Duration of the network period (ms)
    bandwidth: np.array(float)  # Bandwidth of the network period (kbps)
    latency: np.array(float)  # Latency of the network period (ms)


@ModuleOption("trace", default=True)
class TraceClient(Module, DownloadManager):
    log = logging.getLogger("Trace Downloader")

    def __init__(self) -> None:
        """
        Simulate networking conditions using pre-recorded trace files
        """

        super().__init__()
        self.max_packet_size = 20_000

        self.last_time = None  # Last time calling download task
        self.num_periods = None  # Number of network trace periods
        self.trace_idx = None  # Index of the network trace period
        self.rest_time = None  # Time remaining in the current trace period

        self.transfer_queue: asyncio.Queue[tuple[str, bytes | None]] = asyncio.Queue()
        self.content: Dict[str, bytearray] = defaultdict(bytearray)
        self.transfer_size: Dict[str, int] = {}
        self.transfer_compl: Dict[str, asyncio.Event] = {}
        self.downloader_task: Optional[asyncio.Task] = None

    def load_trace(self, trace: str):
        with open(trace) as f:
            duration = []
            bandwidth = []
            latency = []
            for item in json.load(f):
                duration.append(item["duration_ms"])
                bandwidth.append(item["bandwidth_kbps"])
                latency.append(item["latency_ms"])
            self.trace = Trace(np.array(duration), np.array(bandwidth), np.array(latency))
            self.trace_idx = 0
            self.num_periods = self.trace.duration.shape[0]
            self.rest_time = self.trace.duration[self.trace_idx]
        return

    async def setup(self, config: PlayerConfig, **kwargs):
        self.trace = config.trace  # network trace file
        self.log.info(f"Using trace file {self.trace}")
        self.load_trace(self.trace)

        self.downloader_task = asyncio.create_task(self.throttled_download(), name="TASK_LOCAL_DOWNLOADER")

    async def cleanup(self):
        if self.downloader_task:
            self.downloader_task.cancel()

    async def wait_complete(self, url: str) -> Tuple[bytes, int]:
        """
        Wait the stream to complete

        Parameters
        ----------
        url:
            The URL to wait for

        Returns
        -------
            The return value could be None, meaning that the stream got dropped.
            It could be a tuple, the bytes as the first element and size as the second element.
        """
        await self.transfer_compl[url].wait()
        content = self.content[url]
        del self.content[url]
        del self.transfer_compl[url]
        del self.transfer_size[url]
        return content, len(content)

    def cancel_read_url(self, url: str):
        raise NotImplementedError

    async def drop_url(self, url: str):
        """
        Drop the URL downloading process
        """
        raise NotImplementedError

    @property
    def is_busy(self):
        return False

    async def download(self, request: DownloadRequest, save: bool = False) -> Optional[bytes]:
        url = request.url
        self.transfer_compl[url] = asyncio.Event()
        self.transfer_size[url] = Path(url).stat().st_size
        for listener in self.listeners:
            await listener.on_transfer_start(url)
        asyncio.create_task(self.request_read(url), name=f"TASK_LOCAL_REQREAD_{url.rsplit('/', 1)[-1]}")
        if save:
            await self.transfer_compl[url].wait()
            content = self.content[url]
            return content
        else:
            return None


    async def close(self):
        pass

    async def stop(self, url: str):
        pass

    def add_listener(self, listener: DownloadEventListener):
        if listener not in self.listeners:
            self.listeners.append(listener)

    async def request_read(self, url: str):
        # print(f"Request : {url}")
        with open(url, "rb") as f:
            while True:
                data = f.read(self.max_packet_size)
                # print(f"Putting {len(data)} bytes for {url}")
                await self.transfer_queue.put((url, data))
                if not data:
                    break

    async def throttled_download(self):
        while True:
            # print("Getting response from transfer_queue")
            url, chunk = await self.transfer_queue.get()
            if chunk:
                self.content[url].extend(chunk)
                for listener in self.listeners:
                    await listener.on_bytes_transferred(
                        len(chunk), url, len(self.content[url]), self.transfer_size[url], chunk
                    )
                    await asyncio.sleep(self.download_time(len(chunk)))
            else:
                self.transfer_compl[url].set()
                for listener in self.listeners:
                    await listener.on_transfer_end(self.transfer_size[url], url)



    def download_time(self, size: int) -> float:
        # compute elapsed time
        if self.last_time is None:
            self.last_time = time.time()
        else:
            elapsed = time.time() - self.last_time
            self.last_time = time.time()
            self.go_by(elapsed)


        total_time = 0.0
        # transmission latency
        # lat = self.trace.latency[self.trace_idx]
        # while self.rest_time < lat:
        #     lat -= self.rest_time
        #     self.trace_idx = (self.trace_idx + 1) % self.num_periods
        #     self.rest_time = self.trace.duration[self.trace_idx]
        #     total_time += self.rest_time
        # self.rest_time -= lat
        # total_time += lat

        # download latency
        remain = size * 8  # remain download size (bits)
        bw = self.trace.bandwidth[self.trace_idx]
        while remain > bw * self.rest_time:
            remain -= bw * self.rest_time
            self.trace_idx = (self.trace_idx + 1) % self.num_periods
            self.rest_time = self.trace.duration[self.trace_idx]
            bw = self.trace.bandwidth[self.trace_idx]
            total_time += self.rest_time
        self.rest_time -= remain / bw
        total_time += remain / bw

        return total_time / 1000  # convert to seconds

    def go_by(self, time_pass:float):
        # advance timeline
        while self.rest_time < time_pass:
            time_pass -= self.rest_time
            self.trace_idx = (self.trace_idx + 1) % self.num_periods
            self.rest_time = self.trace.duration[self.trace_idx]

        self.rest_time -= time_pass
        return
