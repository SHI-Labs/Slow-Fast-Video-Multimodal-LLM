import av
from av.codec.context import CodecContext
import numpy as np


# This one is faster
def record_video_length_stream(container, indices):
    frames = []
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return frames


# This one works for all types of video
def record_video_length_packet(container):
    frames = []
    # https://github.com/PyAV-Org/PyAV/issues/1269
    # https://www.cnblogs.com/beyond-tester/p/17641872.html
    # context = CodecContext.create("libvpx-vp9", "r")
    for packet in container.demux(video=0):
        for frame in packet.decode():
            frames.append(frame)
    return frames


def read_video_pyav(video_path, num_frm=8):
    container = av.open(video_path)

    if "webm" not in video_path and "mkv" not in video_path:
        # For mp4, we try loading with stream first
        try:
            container = av.open(video_path)
            total_frames = container.streams.video[0].frames
            sampled_frm = min(total_frames, num_frm)
            indices = np.linspace(0, total_frames - 1, sampled_frm, dtype=int)
            frames = record_video_length_stream(container, indices)
        except:
            container = av.open(video_path)
            frames = record_video_length_packet(container)
            total_frames = len(frames)
            sampled_frm = min(total_frames, num_frm)
            indices = np.linspace(0, total_frames - 1, sampled_frm, dtype=int)
            frames = [frames[i] for i in indices]
    else:
        container = av.open(video_path)
        frames = record_video_length_packet(container)
        total_frames = len(frames)
        sampled_frm = min(total_frames, num_frm)
        indices = np.linspace(0, total_frames - 1, sampled_frm, dtype=int)
        frames = [frames[i] for i in indices]
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


def get_frame_indices(total_frames, original_fps, target_fps, num_frm):
    # target_fps = original_fps  # hardcode, should remove after debug
    sample_fps = max(round(original_fps / target_fps), 1)
    frame_idx = [i for i in range(0, total_frames, sample_fps)]
    if len(frame_idx) < num_frm:
        # If we have fewer frames than num_frm, just return all the frames
        return frame_idx
    scale = 1.0 * len(frame_idx) / num_frm
    uniform_idx = [round((i + 1) * scale - 1) for i in range(num_frm)]
    frame_idx = [frame_idx[i] for i in uniform_idx]
    return frame_idx

def read_video_pyav2(video_path, num_frm=8, target_fps=1, threads=8):
    container = av.open(video_path)
    stream = container.streams.video[0]

    stream.thread_type = 'AUTO'
    stream.codec_context.thread_count = threads

    original_fps = stream.average_rate
    total_frames = stream.frames

    if "webm" not in video_path and "mkv" not in video_path:
        try:
            indices = get_frame_indices(total_frames, original_fps, target_fps, num_frm)
            frames = record_video_length_stream(container, indices)
        except:
            container = av.open(video_path)
            frames = record_video_length_packet(container)
            total_frames = len(frames)
            indices = get_frame_indices(total_frames, original_fps, target_fps, num_frm)
            frames = [frames[i] for i in indices]
    else:
        frames = record_video_length_packet(container)
        total_frames = len(frames)
        indices = get_frame_indices(total_frames, original_fps, target_fps, num_frm)
        frames = [frames[i] for i in indices]
    
    total_time = stream.frames / original_fps
    video_info_string = f"Time: {round(total_time, 2)}s; Time interval between frame {round(total_time/len(indices), 3)}s; video tokens:"

    return np.stack([x.to_ndarray(format="rgb24") for x in frames]), video_info_string