"""Advanced speaker clustering — ported from 3D-Speaker (Apache 2.0).

Source references:
    SpectralCluster   : speakerlab/process/cluster.py:23-114
    AHCluster         : speakerlab/process/cluster.py:141-158
    CommonClustering  : speakerlab/process/cluster.py:161-241

Modifications from original:
    - min_pnum=6 added to SpectralCluster (missing in integration plan)
    - UmapHdbscan removed — not needed for meeting diarization, avoids extra deps
    - JointClustering (audio+video) omitted — MeetASR is audio-only
    - Default params tuned for meeting diarization: max_num_spks=15, pval=0.012
"""

from __future__ import annotations

import numpy as np
import scipy.sparse.linalg
from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import squareform
from sklearn.cluster._kmeans import k_means
from sklearn.metrics.pairwise import cosine_similarity


class SpectralCluster:
    """Spectral clustering via unnormalized Laplacian of affinity matrix.

    Adapted from 3D-Speaker cluster.py:23-114 (originally from SpeechBrain).
    Auto-detects number of speakers via eigengap analysis.
    """

    def __init__(
        self,
        min_num_spks: int = 1,
        max_num_spks: int = 15,
        pval: float = 0.012,
        min_pnum: int = 6,
        oracle_num: int | None = None,
    ):
        self.min_num_spks = min_num_spks
        self.max_num_spks = max_num_spks
        self.pval = pval
        self.min_pnum = min_pnum
        self.k = oracle_num

    def __call__(self, X: np.ndarray, **kwargs) -> np.ndarray:
        pval = kwargs.get("pval", None)
        oracle_num = kwargs.get("speaker_num", None)

        # Similarity matrix computation
        sim_mat = self.get_sim_mat(X)

        # Refining similarity matrix with p-pruning
        prunned_sim_mat = self.p_pruning(sim_mat, pval)

        # Symmetrization
        sym_prund_sim_mat = 0.5 * (prunned_sim_mat + prunned_sim_mat.T)

        # Laplacian calculation
        laplacian = self.get_laplacian(sym_prund_sim_mat)

        # Get spectral embeddings
        emb, num_of_spk = self.get_spec_embs(laplacian, oracle_num)

        # Perform clustering
        labels = self.cluster_embs(emb, num_of_spk)

        return labels

    def get_sim_mat(self, X: np.ndarray) -> np.ndarray:
        """Cosine similarity matrix."""
        M = cosine_similarity(X, X)
        return M

    def p_pruning(self, A: np.ndarray, pval: float | None = None) -> np.ndarray:
        """Prune affinity matrix by zeroing out low-similarity entries per row."""
        if pval is None:
            pval = self.pval
        n_elems = int((1 - pval) * A.shape[0])
        n_elems = min(n_elems, A.shape[0] - self.min_pnum)

        # For each row in affinity matrix
        for i in range(A.shape[0]):
            low_indexes = np.argsort(A[i, :])
            low_indexes = low_indexes[0:n_elems]

            # Replace smaller similarity values by 0s
            A[i, low_indexes] = 0
        return A

    def get_laplacian(self, M: np.ndarray) -> np.ndarray:
        """Compute unnormalized Laplacian: L = D - M."""
        M = M.copy()
        np.fill_diagonal(M, 0)
        D = np.sum(np.abs(M), axis=1)

        L = -M
        L[np.diag_indices_from(L)] = D
        return L

    def get_spec_embs(
        self, L: np.ndarray, k_oracle: int | None = None,
    ) -> tuple[np.ndarray, int]:
        """Extract spectral embeddings and estimate number of speakers."""
        if k_oracle is None:
            k_oracle = self.k

        lambdas, eig_vecs = scipy.sparse.linalg.eigsh(
            L, k=min(self.max_num_spks + 1, L.shape[0]), which="SM",
        )

        if k_oracle is not None:
            num_of_spk = k_oracle
        else:
            lambda_gap_list = self._get_eigen_gaps(
                lambdas[self.min_num_spks - 1 : self.max_num_spks + 1]
            )
            num_of_spk = np.argmax(lambda_gap_list) + self.min_num_spks

        emb = eig_vecs[:, :num_of_spk]
        return emb, num_of_spk

    def cluster_embs(self, emb: np.ndarray, k: int) -> np.ndarray:
        """K-means on spectral embeddings."""
        _, labels, _ = k_means(emb, k)
        return labels

    def _get_eigen_gaps(self, eig_vals: np.ndarray) -> list[float]:
        """Compute consecutive eigenvalue gaps for eigengap heuristic."""
        eig_vals_gap_list = []
        for i in range(len(eig_vals) - 1):
            gap = float(eig_vals[i + 1]) - float(eig_vals[i])
            eig_vals_gap_list.append(gap)
        return eig_vals_gap_list


class AHCluster:
    """Agglomerative Hierarchical Clustering with cosine distance.

    Fallback for short segments (< cluster_line). Bottom-up approach that
    iteratively merges closest clusters until cosine threshold is reached.

    Adapted from 3D-Speaker cluster.py:141-158 (originally from VBx).
    """

    def __init__(self, fix_cos_thr: float = 0.4):
        self.fix_cos_thr = fix_cos_thr

    def __call__(self, X: np.ndarray, **kwargs) -> np.ndarray:
        import fastcluster

        scr_mx = cosine_similarity(X)
        scr_mx = squareform(-scr_mx, checks=False)
        lin_mat = fastcluster.linkage(
            scr_mx, method="average", preserve_input="False",
        )
        adjust = abs(lin_mat[:, 2].min())
        lin_mat[:, 2] += adjust
        labels = fcluster(lin_mat, -self.fix_cos_thr + adjust, criterion="distance") - 1
        return labels


class CommonClustering:
    """Orchestrator: selects Spectral or AHC based on segment count.

    Includes post-processing: filter_minor_cluster + merge_by_cos.

    Ported from 3D-Speaker cluster.py:161-241.
    """

    def __init__(
        self,
        cluster_type: str = "spectral",
        cluster_line: int = 40,
        mer_cos: float | None = 0.8,
        min_cluster_size: int = 4,
        **kwargs,
    ):
        self.cluster_type = cluster_type
        self.cluster_line = cluster_line
        self.min_cluster_size = min_cluster_size
        self.mer_cos = mer_cos

        if self.cluster_type == "spectral":
            self.cluster = SpectralCluster(**kwargs)

        elif self.cluster_type == "AHC":
            self.cluster = AHCluster(**kwargs)
        else:
            raise ValueError(f"Cluster type '{self.cluster_type}' is not supported.")

        if self.cluster_type != "AHC":
            self.cluster_for_short = AHCluster()
        else:
            self.cluster_for_short = self.cluster

    def __call__(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """Cluster embeddings and return speaker labels.

        Args:
            X: Embedding matrix of shape [N, D].
            **kwargs: Passed to underlying cluster (e.g. speaker_num).

        Returns:
            Integer label array of length N.
        """
        assert len(X.shape) == 2, "Shape of input should be [N, D]"
        if X.shape[0] <= 1:
            return np.zeros(X.shape[0], dtype=int)

        if X.shape[0] < self.cluster_line:
            labels = self.cluster_for_short(X)
        else:
            labels = self.cluster(X, **kwargs)

        # Remove extremely minor clusters
        labels = self.filter_minor_cluster(labels, X, self.min_cluster_size)
        # Merge similar speakers
        if self.mer_cos is not None:
            labels = self.merge_by_cos(labels, X, self.mer_cos)

        return labels

    def filter_minor_cluster(
        self, labels: np.ndarray, x: np.ndarray, min_cluster_size: int,
    ) -> np.ndarray:
        """Reassign segments in tiny clusters to nearest major cluster."""
        cset = np.unique(labels)
        csize = np.array([(labels == i).sum() for i in cset])
        minor_idx = np.where(csize <= self.min_cluster_size)[0]
        if len(minor_idx) == 0:
            return labels

        minor_cset = cset[minor_idx]
        major_idx = np.where(csize > self.min_cluster_size)[0]
        if len(major_idx) == 0:
            return np.zeros_like(labels)
        major_cset = cset[major_idx]
        major_center = np.stack([x[labels == i].mean(0) for i in major_cset])

        for i in range(len(labels)):
            if labels[i] in minor_cset:
                cos_sim = cosine_similarity(x[i][np.newaxis], major_center)
                labels[i] = major_cset[cos_sim.argmax()]

        return labels

    def merge_by_cos(
        self, labels: np.ndarray, x: np.ndarray, cos_thr: float,
    ) -> np.ndarray:
        """Iteratively merge speaker clusters with cosine similarity > threshold."""
        assert 0 < cos_thr <= 1
        while True:
            cset = np.unique(labels)
            if len(cset) == 1:
                break
            centers = np.stack([x[labels == i].mean(0) for i in cset])
            affinity = cosine_similarity(centers, centers)
            affinity = np.triu(affinity, 1)
            idx = np.unravel_index(np.argmax(affinity), affinity.shape)
            if affinity[idx] < cos_thr:
                break
            c1, c2 = cset[np.array(idx)]
            labels[labels == c2] = c1
        return labels
