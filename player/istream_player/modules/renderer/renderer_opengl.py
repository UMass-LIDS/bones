import asyncio
import logging
import sys
from asyncio import create_subprocess_exec
from asyncio.subprocess import PIPE
from typing import Dict, Union

from istream_player.config.config import PlayerConfig
from istream_player.core.renderer import Renderer
from istream_player.core.downloader import (DownloadEventListener,
                                            DownloadManager)
from istream_player.core.module import Module, ModuleOption
from istream_player.core.mpd_provider import MPDProvider
from istream_player.core.player import Player, PlayerEventListener
from istream_player.models import Segment, State

import pycuda
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from istream_player.modules.decoder import DecoderNvCodec, TensorConverter
import numpy as np
import PyNvCodec as nvc
import time
import torch


class FPSLogger:
    def __init__(self, interval=1):
        self.interval = interval
        self.framecount = 0
        self.seconds = time.time()

    def log(self, titlebar=True, fmt="fps : {0}"):
        self.framecount += 1
        if self.seconds + self.interval < time.time():
            self.fps = self.framecount / self.interval
            self.framecount = 0
            self.seconds = time.time()
            if titlebar:
                glutSetWindowTitle(fmt.format(self.fps))
            else:
                return fmt.format(self.fps)


@ModuleOption("opengl", requires=[Player])
class OpenGLRenderer(Module, Renderer, PlayerEventListener):
    log = logging.getLogger("OpenGL Renderer")

    def __init__(self) -> None:
        super().__init__()
        self.config = None
        self.width = None
        self.height = None
        self.fps = None
        self.data = None
        self.device = None

        self.nv_down = None
        self.nv_up = None
        self.last_render_time = None

        self.fps_logger = FPSLogger()

        self.task_start = None
        self.task_total = None

        self.tensor_converter = None

    async def setup(self, config: PlayerConfig, player: Player):
        player.add_listener(self)
        self.config = config
        self.width, self.height = config.display_width, config.display_height
        self.fps = config.display_fps
        self.device = config.renderer_device

        self.nv_down = nvc.PySurfaceDownloader(self.width, self.height, nvc.PixelFormat.RGB, 0)
        self.nv_up = nvc.PyFrameUploader(self.width, self.height, nvc.PixelFormat.RGB, 0)
        self.data = np.zeros((self.width * self.height, 3), np.uint8)

        self.tensor_converter = TensorConverter(self.width, self.height, self.width, self.height, 0)

    def remain_task(self):
        if self.task_start is None:
            return 0
        return max(self.task_total - (time.time() - self.task_start), 0)


    async def on_segment_playback_start(self, segments: Dict[int, Segment]):

        # First time rendering
        if self.last_render_time is None:
            self.setup_display(self.width, self.height)
            self.setup_opengl()
            self.create_textures()
            # self.cuda_gl_handshake()

        for as_idx in segments:
            segment = segments[as_idx]
            self.task_start = time.time()
            self.task_total = segment.duration
            self.log.info(f"Rendering segment {segment.url}")
            if segment.decode_data is None:
                decoder = DecoderNvCodec(self.config, segment, resize=True)
                self.cuda_gl_handshake()

                while True:
                    surf = decoder.decode_one_frame()

                    if surf is None:
                        break

                    if self.last_render_time is None:
                        self.last_render_time = time.time()

                    self.render_one_frame(surf)
                    # Next render time
                    next_render_time = self.last_render_time + 1 / self.fps
                    sleep_time = max(next_render_time - time.time(), 0)
                    await asyncio.sleep(sleep_time)
                    self.last_render_time = next_render_time
            else:
                for surf in segment.decode_data:
                    if self.last_render_time is None:
                        self.last_render_time = time.time()

                    self.render_one_frame(surf)
                    # Next render time
                    next_render_time = self.last_render_time + 1 / self.fps
                    sleep_time = max(next_render_time - time.time(), 0)
                    await asyncio.sleep(sleep_time)
                    self.last_render_time = next_render_time


    def setup_display(self, width, height):
        self.log.info(f"Setting up display {width}x{height}")
        glutInit(sys.argv)
        glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
        glutInitWindowSize(width, height)
        glutInitWindowPosition(0, 0)
        glutCreateWindow(b"iStream Player")
        self.log.info(f"Finished setting up display {width}x{height}")

    def setup_opengl(self):
        self.program = self.compile_shaders()
        import pycuda.autoinit
        import pycuda.gl.autoinit

        self.vao = GLuint()
        glCreateVertexArrays(1, self.vao)

    def create_textures(self):
        ## create texture for GL display
        self.texture = glGenTextures(1)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        glTexImage2D(
            GL_TEXTURE_2D,
            0,
            GL_RGB,
            self.width,
            self.height,
            0,
            GL_RGB,
            GL_UNSIGNED_BYTE,
            ctypes.c_void_p(0),
        )
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glBindTexture(GL_TEXTURE_2D, 0)
        self.cuda_img = pycuda.gl.RegisteredImage(
            int(self.texture), GL_TEXTURE_2D, pycuda.gl.graphics_map_flags.NONE
        )  # WRITE_DISCARD)

    def cuda_gl_handshake(self):
        self.pbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.pbo)
        glBufferData(GL_ARRAY_BUFFER, self.data, GL_DYNAMIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        import pycuda.autoinit
        import pycuda.gl.autoinit

        self.cuda_pbo = pycuda.gl.RegisteredBuffer(int(self.pbo))
        self.vao = 0
        glGenVertexArrays(1, self.vao)
        glBindVertexArray(self.vao)

    def compile_shaders(self):
        vertex_shader_source = """
        #version 450 core
        out vec2 uv;
        void main( void)
        {
            // Declare a hard-coded array of positions
            const vec2 vertices[4] = vec2[4](vec2(-0.5,  0.5),
                                                 vec2( 0.5,  0.5),
                                                 vec2( 0.5, -0.5),
                                                 vec2(-0.5, -0.5));
            // Index into our array using gl_VertexID
            uv=vertices[gl_VertexID]+vec2(0.5,0.5);
            uv.y = 1.0 - uv.y; // Flip the texture vertically
            gl_Position = vec4(2*vertices[gl_VertexID],1.0,1.0);
            }
        """

        vertex_shader = glCreateShader(GL_VERTEX_SHADER)
        glShaderSource(vertex_shader, vertex_shader_source)
        glCompileShader(vertex_shader)

        fragment_shader_source = """
        #version 450 core
        uniform sampler2D s;
        in vec2 uv;
        out vec4 color;
        void main(void)
        {
            color = vec4(texture(s, uv));
        }
        """

        fragment_shader = glCreateShader(GL_FRAGMENT_SHADER)
        glShaderSource(fragment_shader, fragment_shader_source)
        glCompileShader(fragment_shader)

        program = glCreateProgram()
        glAttachShader(program, vertex_shader)
        glAttachShader(program, fragment_shader)
        glLinkProgram(program)
        # --- Clean up now that we don't need these shaders anymore.
        glDeleteShader(vertex_shader)
        glDeleteShader(fragment_shader)

        return program

    def to_device(self, surf : nvc.Surface | torch.Tensor):
        # raw frame
        if isinstance(surf, nvc.Surface):
            return surf

        # enhanced frame
        if isinstance(surf, torch.Tensor):
            surf = surf.cuda()  # move to GPU
            surf = self.tensor_converter.tensor_to_surface(surf)
            return surf

    def render_one_frame(self, surf: nvc.Surface | torch.Tensor):
        # glClearBufferfv(GL_COLOR, 0, (0, 0, 0))
        # ## bind program
        # glUseProgram(self.program)

        surf = self.to_device(surf)

        ## texture update through cpu and system memory
        if self.device == "cpu":
            ## Download surface data to CPU, then update GL texture with these data
            success = self.nv_down.DownloadSingleSurface(surf, self.data)
            if not success:
                self.log.info("Could not download Cuda Surface to CPU")
                return

            glClearBufferfv(GL_COLOR, 0, (0, 0, 0))
            ## bind program
            glUseProgram(self.program)

            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, self.texture)
            glTexSubImage2D(
                GL_TEXTURE_2D,
                0,
                0,
                0,
                self.width,
                self.height,
                GL_RGB,
                GL_UNSIGNED_BYTE,
                self.data,
            )
        else:
            glClearBufferfv(GL_COLOR, 0, (0, 0, 0))
            ## bind program
            glUseProgram(self.program)

            ## cuda copy from surface.Plane_Ptr() to pbo, then update texture from PBO
            src_plane = surf.PlanePtr()
            buffer_mapping = self.cuda_pbo.map()
            buffptr, buffsize = buffer_mapping.device_ptr_and_size()
            cpy = pycuda.driver.Memcpy2D()
            cpy.set_src_device(src_plane.GpuMem())
            cpy.set_dst_device(buffptr)
            cpy.width_in_bytes = src_plane.Width()
            cpy.src_pitch = src_plane.Pitch()
            cpy.dst_pitch = self.width * 3
            cpy.height = src_plane.Height()
            cpy(aligned=True)
            # pycuda.driver.Context.synchronize() ## not required?
            buffer_mapping.unmap()
            ## opengl update texture from pbo
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, self.texture)
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, int(self.pbo))
            glTexSubImage2D(
                GL_TEXTURE_2D,
                0,
                0,
                0,
                self.width,
                self.height,
                GL_RGB,
                GL_UNSIGNED_BYTE,
                ctypes.c_void_p(0),
            )

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        ## send uniforms to program and draw quad
        glUniform1i(glGetUniformLocation(self.program, b"s"), 0)
        glDrawArrays(GL_QUADS, 0, 4)
        ## Display
        glutSwapBuffers()

        self.fps_logger.log()
