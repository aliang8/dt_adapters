import numpy as np
from PIL import Image
import wandb
import matplotlib.pyplot as plt
import seaborn as sns


def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n))


def create_video_grid(videos, max_columns=5):
    # wandb needs videos to be in BxCxHxW
    n_frames, height, width, channel = videos[0].shape

    if len(videos) % max_columns != 0:
        # need to pad with some black videos
        extra_videos_needed = ((len(videos) // max_columns) + 1) * max_columns - len(
            videos
        )
        for i in range(extra_videos_needed):
            videos.append(np.zeros_like(videos[0]))

    assert len(videos) % max_columns == 0

    max_seq_length = max([video.shape[0] for video in videos])

    # first resize videos and pad them to max length
    for i, video in enumerate(videos):
        all_frames = []
        for frame in video:
            if frame.shape[0] == 1:
                frame = (
                    frame.reshape((frame.shape[1], frame.shape[2], 1)).repeat(
                        3, axis=-1
                    )
                    * 256
                ).astype(np.uint8)
            frame = Image.fromarray(frame)
            frame = np.array(frame)
            # frame = np.array(frame.resize((height, width)))
            all_frames.append(frame)
        all_frames = np.array(all_frames).transpose(0, 3, 1, 2)
        video = all_frames

        if video.shape[0] < max_seq_length:
            padded_video = np.zeros((max_seq_length, *all_frames.shape[1:]))
            padded_video[: video.shape[0]] = video
            videos[i] = padded_video
        else:
            videos[i] = video

    # split videos into different rows
    num_rows = int(len(videos) / max_columns)
    chunks = list(split(videos, num_rows))

    rows = []
    for chunk in chunks:
        # stick the videos into a grid and concatenate the videos on width
        row = np.concatenate(chunk, axis=-1)
        rows.append(row)

    videos = np.concatenate(rows, axis=-2)  # concat over height
    return videos


def save_videos_to_wandb(videos, task_name="", step=0, fps=10):
    # create grid and log to wandb
    video_array = create_video_grid(videos)
    wandb.log(
        {
            f"eval/{task_name}/rollout_videos": wandb.Video(
                video_array,
                caption=f"train_iter_{step}",
                fps=fps,
                format="gif",
            )
        }
    )


def visualize_fusion_attention(adapter_fusion_attentions, n_layers=4):
    # the adapter fusion attention is of size [bs, seq_len, n_tasks]
    # get layer-wise attention scores and log as heatmap
    attn_matrix = None
    attn_scores = adapter_fusion_attentions

    key = list(attn_scores.keys())[0]

    for idx in range(n_layers):
        layer_weights = attn_scores[key][idx]["output_adapter"][np.newaxis, :]

        if attn_matrix is None:
            attn_matrix = layer_weights
        else:
            attn_matrix = np.concatenate([attn_matrix, layer_weights], axis=0)

    # average over first and second dimension
    # should be of size [n_layers, n_tasks]
    attn_matrix = attn_matrix.mean(axis=1).mean(axis=1)

    plt.clf()
    ax = sns.heatmap(data=attn_matrix, vmin=0, vmax=1, annot=True, cmap="YlGnBu")
    fig = ax.figure

    # need to do this weird hack to first convert to PIL Image
    img = Image.frombytes(
        "RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb()
    )
    return img
