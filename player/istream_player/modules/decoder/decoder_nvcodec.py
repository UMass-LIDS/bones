import sys
import os
import tqdm
import torch
import torchvision.transforms as T
import PyNvCodec as nvc
try:
    import PytorchNvCodec as pnvc
except ImportError as err:
    raise (f"""Could not import `PytorchNvCodec`: {err}.
Please make sure it is installed! Run
`pip install git+https://github.com/NVIDIA/VideoProcessingFramework#subdirectory=src/PytorchNvCodec` or
`pip install src/PytorchNvCodec` if using a local copy of the VideoProcessingFramework repository""")  # noqa

from istream_player.models import Segment
from istream_player.config.config import PlayerConfig
import numpy as np
import tempfile
from typing import Optional, Tuple


class Converter:
    def __init__(self, width: int, height: int, gpu_id: int, resize: bool = False):
        self.gpu_id = gpu_id
        self.resize = resize

        self.resizer = nvc.PySurfaceResizer(width, height, nvc.PixelFormat.NV12, gpu_id)
        self.to_yuv = nvc.PySurfaceConverter(width, height, nvc.PixelFormat.NV12, nvc.PixelFormat.YUV420, gpu_id)
        self.to_rgb = nvc.PySurfaceConverter(width, height, nvc.PixelFormat.YUV420, nvc.PixelFormat.RGB, gpu_id)

        # self.resizer = nvc.PySurfaceResizer(display_H, display_W, nvc.PixelFormat.RGB, gpu_id)
        self.context = nvc.ColorspaceConversionContext(nvc.ColorSpace.BT_601, nvc.ColorRange.MPEG)

    def run(self, src_surface: nvc.Surface) -> nvc.Surface:
        surf = src_surface
        if self.resize:
            surf = self.resizer.Execute(surf)
        surf = self.to_yuv.Execute(surf, self.context)
        surf = self.to_rgb.Execute(surf, self.context)
        return surf


class DecoderNvCodec():
    def __init__(self, config: PlayerConfig, segment: Segment, gpu_id: int = 0, resize: bool = False):
        self.config = config
        self.segment = segment
        self.gpu_id = gpu_id

        mp4_path = self.m4s_to_mp4()
        self.nv_dec = nvc.PyNvDecoder(mp4_path, gpu_id)

        if resize:
            width = config.display_width
            height = config.display_height
        else:
            width = self.nv_dec.Width()
            height = self.nv_dec.Height()
        self.converter = Converter(width, height, gpu_id, resize=resize)
        return

    def resolution(self) -> Tuple[int, int]:
        return self.nv_dec.Width(), self.nv_dec.Height()

    def num_frames(self) -> int:
        return self.nv_dec.Numframes()

    def decode_one_frame(self) -> Optional[nvc.Surface]:
        surf = self.nv_dec.DecodeSingleSurface()
        if surf.Empty():
            return None
        surf = self.converter.run(surf)
        # return surf.Clone(self.gpu_id)
        return surf

    def m4s_to_mp4(self):
        """
        Concatenate the stream initialization and a m4s video chunk into a mp4 file.
        """
        def cat(input_file, output_file):
            with open(input_file, 'rb') as infile, open(output_file, 'ab') as outfile:
                # Read the contents of the input file
                content = infile.read()
                # Append the contents to the output file
                outfile.write(content)

        mp4_path = tempfile.NamedTemporaryFile(dir=self.config.run_dir, delete=False, suffix=".mp4").name
        init_path = self.segment.init_path
        seg_path = self.segment.path
        cat(init_path, mp4_path)
        cat(seg_path, mp4_path)
        return mp4_path


class TensorConverter:
    def __init__(self, decode_W: int, decode_H: int, display_W: int, display_H: int, gpu_id: int = 0):
        self.decode_W = decode_W
        self.decode_H = decode_H
        self.display_W = display_W
        self.display_H = display_H

        self.gpu_id = gpu_id
        self.to_planar = nvc.PySurfaceConverter(decode_W, decode_H, nvc.PixelFormat.RGB, nvc.PixelFormat.RGB_PLANAR, self.gpu_id)
        self.to_rgb = nvc.PySurfaceConverter(display_W, display_H, nvc.PixelFormat.RGB_PLANAR, nvc.PixelFormat.RGB, self.gpu_id)
        self.context = nvc.ColorspaceConversionContext(nvc.ColorSpace.BT_601, nvc.ColorRange.MPEG)

    def surface_to_tensor(self, surface: nvc.Surface) -> torch.Tensor:
        """
        Converts planar rgb surface to cuda float tensor.

        Args:
            surface: planar rgb surface

        Returns:
            cuda float tensor of shape (1, 3, height, width)
        """
        surface = self.to_planar.Execute(surface, self.context)
        surface = surface.Clone()

        surf_plane = surface.PlanePtr()
        img_tensor = pnvc.DptrToTensor(
            surf_plane.GpuMem(),
            surf_plane.Width(),
            surf_plane.Height(),
            surf_plane.Pitch(),
            surf_plane.ElemSize(),
        )
        if img_tensor is None:
            raise RuntimeError("Can not export to tensor.")

        img_tensor.resize_(3, int(surf_plane.Height() / 3), surf_plane.Width())
        img_tensor = img_tensor.type(dtype=torch.cuda.FloatTensor)
        img_tensor = torch.divide(img_tensor, 255.0)
        img_tensor = torch.clamp(img_tensor, 0.0, 1.0)

        img_tensor = torch.unsqueeze(img_tensor, 0)
        # return img_tensor.clone()
        return img_tensor

    def tensor_to_surface(self, img_tensor: torch.tensor, gpu_id: int = 0) -> nvc.Surface:
        """
        Converts cuda float tensor to planar rgb surface.

        Args:
            img_tensor: cuda float tensor of shape (1, 3, height, width)
            gpu_id: gpu id to allocate the surface on

        Returns:
            planar rgb surface
        """
        img_tensor = torch.squeeze(img_tensor, 0)
        img = torch.clamp(img_tensor, 0.0, 1.0)
        img = torch.multiply(img, 255.0).contiguous()
        img = img.type(dtype=torch.cuda.ByteTensor).clone()

        surface = nvc.Surface.Make(nvc.PixelFormat.RGB_PLANAR, self.display_W, self.display_H, gpu_id)
        surf_plane = surface.PlanePtr()
        pnvc.TensorToDptr(
            img,
            surf_plane.GpuMem(),
            surf_plane.Width(),
            surf_plane.Height(),
            surf_plane.Pitch(),
            surf_plane.ElemSize(),
        )
        surface = self.to_rgb.Execute(surface, self.context)
        # return surface.Clone(self.gpu_id)
        return surface