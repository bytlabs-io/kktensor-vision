import asyncio
import logging
import time
import cv2
import numpy as np
import modules.globals
from PIL import Image, ImageOps
import modules.metadata

from dotenv import load_dotenv

from livekit import rtc
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    WorkerOptions,
    cli,
)
from modules import core
from modules.face_analyser import (
    get_one_face,
)
from modules.processors.frame.core import get_frame_processors_modules

load_dotenv()
logger = logging.getLogger("echo-agent")

# An example agent that echos each utterance from the user back to them
# the example uses a queue to buffer incoming streams, and uses VAD to detect
# when the user is done speaking.

async def create_webcam_preview(frame) -> rtc.VideoFrame:
    frame_processors = get_frame_processors_modules(modules.globals.frame_processors)
    source_image = None
    prev_time = time.time()
    fps_update_interval = 0.5
    frame_count = 0
    fps = 0

    temp_frame = np.asarray(frame).copy()

    if modules.globals.live_mirror:
        temp_frame = cv2.flip(temp_frame, 1)

    # Perform resize on frame

    if source_image is None and modules.globals.source_path:
        source_image = get_one_face(cv2.imread(modules.globals.source_path))

    for frame_processor in frame_processors:
        if frame_processor.NAME == "DLC.FACE-ENHANCER":
            if modules.globals.fp_ui["face_enhancer"]:
                temp_frame = frame_processor.process_frame(None, temp_frame)
        else:
            temp_frame = frame_processor.process_frame(source_image, temp_frame)

    # Calculate and display FPS
    current_time = time.time()
    frame_count += 1
    if current_time - prev_time >= fps_update_interval:
        fps = frame_count / (current_time - prev_time)
        frame_count = 0
        prev_time = current_time

    if modules.globals.show_fps:
        cv2.putText(
            temp_frame,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

    # image = cv2.cvtColor(temp_frame, cv2.COLOR_BGR2RGB)
    # image = Image.fromarray(image)
    # image = ImageOps.contain(
    #     image, (temp_frame.shape[1], temp_frame.shape[0]), Image.LANCZOS
    # )
    # convert image to byte array
    # image = bytearray(np.asarray(image))

    # convert temp_frame to livekit VideoFrame

    return temp_frame


async def entrypoint(ctx: JobContext):
    logger.info(f"connecting to room {ctx.room.name}")
    await ctx.connect(auto_subscribe=AutoSubscribe.SUBSCRIBE_ALL)

    # wait for the first participant to connect
    participant: rtc.Participant = await ctx.wait_for_participant()
    a_stream = rtc.AudioStream.from_participant(
        participant=participant,
        track_source=rtc.TrackSource.SOURCE_MICROPHONE,
    )

    v_stream = rtc.VideoStream.from_participant(
        participant=participant,
        track_source=rtc.TrackSource.SOURCE_CAMERA,
    )
    # get widht and height of the video stream from first frame
    first_frame = await v_stream.__anext__()
    width = first_frame.width
    height = first_frame.height

    a_source = rtc.AudioSource(sample_rate=48000, num_channels=1)
    v_source = rtc.VideoSource(width=width, height=height)
    a_track = rtc.LocalAudioTrack.create_audio_track("echo-a", a_source)
    v_track = rtc.LocalVideoTrack.create_video_track("echo-v", v_source)
    await ctx.room.local_participant.publish_track(
        a_track,
        rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE),
    )
    await ctx.room.local_participant.publish_track(
        v_track,
        rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_CAMERA),
    )

    async def _process_audio():
        async for ev in a_stream:
            # delay for 100ms fo It can sync with video, this should be programmable for the user
            # await asyncio.sleep(0.2)
            await a_source.capture_frame(ev.frame)

    async def _process_video():
        async for ev in v_stream:
            data = await create_webcam_preview(ev.frame.data)
            # save frame as image
            frame = rtc.VideoFrame(width=ev.frame.width, height=ev.frame.height, type=rtc.VideoBufferType.RGBA, data=data)
            # cv2.imwrite(f"media/frame_{time.time().__str__}.jpg", data)
            print(frame)
            v_source.capture_frame(frame)


    await asyncio.gather(
        _process_video(),
        _process_audio()
    )

# if __name__ == '__main__':
#     core.run()

if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
        ),
    )