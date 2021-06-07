import numpy as np
import matplotlib.pyplot as plt


def plotclusters3D(data, labels, peaks):
    """
    Plots the modes of the given image data in 3D by coloring each pixel
    according to its corresponding peak.

    Args:
        data: image data in the format [number of pixels]x[feature vector].
        labels: a list of labels, one for each pixel.
        peaks: a list of vectors, whose first three components can
        be interpreted as BGR values.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    bgr_peaks = np.array(peaks[:, 0:2], dtype=float)
    rgb_peaks = bgr_peaks[...,::-1]
    rgb_peaks /= 255.0

    peaks_it = np.array(peaks[:, 0:2].T)

    for idx, peak in enumerate(peaks_it):
        inds = np.where(labels == idx + 1)
        cluster = data[:, inds[0]].T

        ax.scatter(cluster[:, 0], cluster[:, 1], cluster[:, 2], color='y', marker="*", s=.5, alpha=0.4)
        ax.scatter(peak[0], peak[1], peak[2], color='b', marker="x", s=110)

    plt.show()


def plotclusters3D_rgb(data, labels, peaks):
    """
        Plots the modes of the given image data in 3D by coloring each pixel
        according to its corresponding peak.

        Args:
            data: image data in the format [number of pixels]x[feature vector].
            labels: a list of labels, one for each pixel.
            peaks: a list of vectors, whose first three components can
            be interpreted as BGR values.
        """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    bgr_peaks = np.array(peaks, dtype=float)
    rgb_peaks = bgr_peaks[..., ::-1]
    rgb_peaks /= 255.0

    colors = ["g", "y", "c", "b", "k"]

    for idx, peak in enumerate(peaks):
        inds = np.where(labels == idx + 1)
        cluster = data[:, inds[0]].T

        ax.scatter(cluster[:, 0], cluster[:, 1], cluster[:, 2], colors, marker="*")

    plt.show()
