from typing import Union, Dict, Optional
import os
import math
import numbers
import zarr
import numcodecs
import numpy as np
from functools import cached_property


def check_chunks_compatible(chunks: tuple, shape: tuple):
    assert len(shape) == len(chunks)
    for c in chunks:
        assert isinstance(c, numbers.Integral)
        assert c > 0


def rechunk_recompress_array(
    group, name, chunks=None, chunk_length=None, compressor=None, tmp_key="_temp"
):
    old_arr = group[name]
    if chunks is None:
        if chunk_length is not None:
            chunks = (chunk_length,) + old_arr.chunks[1:]
        else:
            chunks = old_arr.chunks
    check_chunks_compatible(chunks, old_arr.shape)

    if compressor is None:
        compressor = old_arr.compressor

    if (chunks == old_arr.chunks) and (compressor == old_arr.compressor):
        # no change
        return old_arr

    # rechunk recompress
    group.move(name, tmp_key)
    old_arr = group[tmp_key]
    n_copied, n_skipped, n_bytes_copied = zarr.copy(
        source=old_arr,
        dest=group,
        name=name,
        chunks=chunks,
        compressor=compressor,
    )
    del group[tmp_key]
    arr = group[name]
    return arr


def get_optimal_chunks(shape, dtype, target_chunk_bytes=2e6, max_chunk_length=None):
    """
    Common shapes
    T,D
    T,N,D
    T,H,W,C
    T,N,H,W,C
    """
    itemsize = np.dtype(dtype).itemsize
    # reversed
    rshape = list(shape[::-1])
    if max_chunk_length is not None:
        rshape[-1] = int(max_chunk_length)
    # step 1. find which dims, when grouped, is still smaller than the target size
    split_idx = len(shape) - 1
    for i in range(len(shape) - 1):
        this_chunk_bytes = itemsize * np.prod(rshape[:i])
        next_chunk_bytes = itemsize * np.prod(rshape[: i + 1])
        if (
            this_chunk_bytes <= target_chunk_bytes
            and next_chunk_bytes > target_chunk_bytes
        ):
            split_idx = i
    # step 2. find if the rest dims still need to be further divided
    # e.g. shape = (1000, 5), target size is 16. Then chunk should be (3, 5), instead of (5)
    rchunks = rshape[:split_idx]
    item_chunk_bytes = itemsize * np.prod(rshape[:split_idx])
    this_max_chunk_length = rshape[split_idx]
    next_chunk_length = min(
        this_max_chunk_length, math.ceil(target_chunk_bytes / item_chunk_bytes)
    )
    rchunks.append(next_chunk_length)  # desired chunk size!
    # step 3. clean up: match chunk shape dimension, reverse order back
    len_diff = len(shape) - len(rchunks)
    rchunks.extend([1] * len_diff)  # add len_diff dimensions
    chunks = tuple(rchunks[::-1])
    # print(np.prod(chunks) * itemsize / target_chunk_bytes)
    return chunks


class ReplayBuffer:
    """
    Zarr-based temporal datastructure.
    Assumes first dimension to be time. Only chunk in time dimension.
    """

    def __init__(self, root: Union[zarr.Group, Dict[str, dict]]):
        """
        Dummy constructor. Use copy_from* and create_from* class methods instead.
        """
        assert "data" in root
        assert "meta" in root
        assert "episode_robot0_len" in root["meta"]
        self.root = root

    # ============= create constructors ===============
    @classmethod
    def create_empty_zarr(cls, storage=None, root=None):
        if root is None:
            if storage is None:
                storage = zarr.MemoryStore()
            root = zarr.group(store=storage)
        data = root.require_group("data", overwrite=False)
        meta = root.require_group("meta", overwrite=False)
        # if "episode_rgb0_len" not in meta:
        #     episode_rgb0_len = meta.zeros(
        #         "episode_rgb0_len",
        #         shape=(0,),
        #         dtype=np.int64,
        #         compressor=None,
        #         overwrite=False,
        #     )
        return cls(root=root)

    # ============= copy constructors ===============
    @classmethod
    def copy_from_store(
        cls,
        src_store,
        dest_store=None,
        keys=None,
        chunks: Dict[str, tuple] = dict(),
        compressors: Union[dict, str, numcodecs.abc.Codec] = dict(),
        if_exists="replace",
        **kwargs,
    ):
        """
        Load to memory.
        """
        src_root = zarr.group(src_store)
        
        if dest_store is None:
            print("Error: need to provide a store.")
            exit(-1)
        dest_root = zarr.group(store=dest_store)
        # dest_data_group = dest_root.create_group('data', overwrite=True)

        if "data" in src_root:
            assert "meta" in src_root
            src_data_group = src_root["data"]
            print("[ReplayBuffer] checking chunk size and compressor.")
            passed = True
            for ep in src_data_group.keys():
                # iterates over episodes
                # ep: 'episode_xx'
                print(" checking: ", ep)
                # dest_data_group.create_group(ep)
                keys = src_data_group[ep].keys()
                for key in keys:
                    key_path = f"/data/{ep}/{key}"
                    # print('loading key: ', key)
                    value = src_data_group[ep][key]
                    cks = cls._resolve_array_chunks(chunks=chunks, key=key, array=value)
                    cpr = cls._resolve_array_compressor(
                        compressors=compressors, key=key, array=value
                    )
                    key_path = f"/data/{ep}/{key}"
                    if cks != value.chunks or cpr != value.compressor:
                        print(
                            f"[ReplayBuffer] input array failed chunk&compressor check for {key_path}."
                        )
                        print(
                            f"[ReplayBuffer]  input chunk: {value.chunks}, recommended chunk: {cks}"
                        )
                        print(
                            f"[ReplayBuffer]  input compressor: {value.compressor}, recommended compressor: {cpr}"
                        )
            if passed:
                print("[ReplayBuffer] copying data to memory store.")
                n_copied, n_skipped, n_bytes_copied = zarr.copy_store(
                    source=src_store, dest=dest_store,
                    if_exists=if_exists
                )
                print("copied to replaybuffer: ", n_copied, n_skipped, n_bytes_copied)
                dest_root = zarr.group(store=dest_store)
            else:
                exit(-1)
        else:
            print("[ReplayBuffer] copy_from_store: no data found in the source store.")
            exit(-1)

        buffer = cls(root=dest_root)
        return buffer

    @classmethod
    def copy_from_path(
        cls,
        zarr_path,
        store=None,
        keys=None,
        chunks: Dict[str, tuple] = dict(),
        compressors: Union[dict, str, numcodecs.abc.Codec] = dict(),
        if_exists="replace",
        **kwargs,
    ):
        """
        Copy a on-disk zarr to in-memory compressed.
        Recommended
        """
        group = zarr.open(os.path.expanduser(zarr_path), "r")
        return cls.copy_from_store(
            src_store=group.store,
            store=store,
            keys=keys,
            chunks=chunks,
            compressors=compressors,
            if_exists=if_exists,
            **kwargs,
        )

    # ============= save methods ===============
    def save_to_store(
        self,
        store,
        chunks: Optional[Dict[str, tuple]] = dict(),
        compressors: Union[str, numcodecs.abc.Codec, dict] = dict(),
        if_exists="replace",
        **kwargs,
    ):

        root = zarr.group(store)
        if self.backend == "zarr":
            # recompression free copy
            n_copied, n_skipped, n_bytes_copied = zarr.copy_store(
                source=self.root.store,
                dest=store,
                source_path="/meta",
                dest_path="/meta",
                if_exists=if_exists,
            )
        else:
            meta_group = root.create_group("meta", overwrite=True)
            # save meta, no chunking
            for key, value in self.root["meta"].items():
                _ = meta_group.array(
                    name=key, data=value, shape=value.shape, chunks=value.shape
                )

        # save data, chunk
        data_group = root.create_group("data", overwrite=True)
        for key, value in self.root["data"].items():
            cks = self._resolve_array_chunks(chunks=chunks, key=key, array=value)
            cpr = self._resolve_array_compressor(
                compressors=compressors, key=key, array=value
            )
            if isinstance(value, zarr.Array):
                if cks == value.chunks and cpr == value.compressor:
                    # copy without recompression
                    this_path = "/data/" + key
                    n_copied, n_skipped, n_bytes_copied = zarr.copy_store(
                        source=self.root.store,
                        dest=store,
                        source_path=this_path,
                        dest_path=this_path,
                        if_exists=if_exists,
                    )
                else:
                    # copy with recompression
                    n_copied, n_skipped, n_bytes_copied = zarr.copy(
                        source=value,
                        dest=data_group,
                        name=key,
                        chunks=cks,
                        compressor=cpr,
                        if_exists=if_exists,
                    )
            else:
                # numpy
                _ = data_group.array(name=key, data=value, chunks=cks, compressor=cpr)
        return store

    def save_to_path(
        self,
        zarr_path,
        chunks: Optional[Dict[str, tuple]] = dict(),
        compressors: Union[str, numcodecs.abc.Codec, dict] = dict(),
        if_exists="replace",
        **kwargs,
    ):
        store = zarr.DirectoryStore(os.path.expanduser(zarr_path))
        return self.save_to_store(
            store, chunks=chunks, compressors=compressors, if_exists=if_exists, **kwargs
        )

    @staticmethod
    def resolve_compressor(compressor="default"):
        if compressor == "default":
            compressor = numcodecs.Blosc(
                cname="lz4", clevel=5, shuffle=numcodecs.Blosc.NOSHUFFLE
            )
        elif compressor == "disk":
            compressor = numcodecs.Blosc(
                "zstd", clevel=5, shuffle=numcodecs.Blosc.BITSHUFFLE
            )
        return compressor

    @classmethod
    def _resolve_array_compressor(
        cls, compressors: Union[dict, str, numcodecs.abc.Codec], key, array
    ):
        # allows compressor to be explicitly set to None
        cpr = "nil"
        if isinstance(compressors, dict):
            if key in compressors:
                cpr = cls.resolve_compressor(compressors[key])
            elif isinstance(array, zarr.Array):
                cpr = array.compressor
        else:
            cpr = cls.resolve_compressor(compressors)
        # backup default
        if cpr == "nil":
            cpr = cls.resolve_compressor("default")
        return cpr

    @classmethod
    def _resolve_array_chunks(cls, chunks: Union[dict, tuple], key, array):
        cks = None
        if isinstance(chunks, dict):
            if key in chunks:
                cks = chunks[key]
            elif isinstance(array, zarr.Array):
                cks = array.chunks
        elif isinstance(chunks, tuple):
            cks = chunks
        else:
            raise TypeError(f"Unsupported chunks type {type(chunks)}")
        # backup default
        if cks is None:
            cks = get_optimal_chunks(shape=array.shape, dtype=array.dtype)
        # check
        check_chunks_compatible(chunks=cks, shape=array.shape)
        return cks

    # ============= properties =================
    @cached_property
    def data(self):
        return self.root["data"]

    @cached_property
    def meta(self):
        return self.root["meta"]

    def update_meta(self, data):
        # sanitize data
        np_data = dict()
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                np_data[key] = value
            else:
                arr = np.array(value)
                if arr.dtype == object:
                    raise TypeError(f"Invalid value type {type(value)}")
                np_data[key] = arr

        meta_group = self.meta
        if self.backend == "zarr":
            for key, value in np_data.items():
                _ = meta_group.array(
                    name=key,
                    data=value,
                    shape=value.shape,
                    chunks=value.shape,
                    overwrite=True,
                )
        else:
            meta_group.update(np_data)

        return meta_group

    @property
    def episode_rgb0_len(self):
        return self.meta["episode_rgb0_len"]

    @property
    def episode_robot0_len(self):
        return self.meta["episode_robot0_len"]

    @property
    def backend(self):
        backend = "numpy"
        if isinstance(self.root, zarr.Group):
            backend = "zarr"
        return backend

    # =========== dict-like API ==============
    def __repr__(self) -> str:
        if self.backend == "zarr":
            return str(self.root.tree())
        else:
            return super().__repr__()

    def keys(self):
        return self.root.keys()

    def values(self):
        return self.root.values()

    def items(self):
        return self.root.items()

    def __getitem__(self, key):
        return self.root[key]

    # =========== our API ==============
    @property
    def n_steps(self):
        raise NotImplementedError
        # this is not used
        if len(self.episode_robot_len) == 0:
            return 0
        return np.sum(self.episode_robot_len)

    @property
    def n_episodes(self):
        return len(self.episode_robot0_len)

    @property
    def chunk_size(self):
        if self.backend == "zarr":
            return next(iter(self.data.arrays()))[-1].chunks[0]
        return None

    @property
    def episode_lengths(self):
        raise NotImplementedError
        # this is not used
        return self.episode_robot_len[:]

    # =========== chunking =============
    def get_chunks(self) -> dict:
        assert self.backend == "zarr"
        chunks = dict()
        for key, value in self.data.items():
            chunks[key] = value.chunks
        return chunks

    def set_chunks(self, chunks: dict):
        assert self.backend == "zarr"
        for key, value in chunks.items():
            if key in self.data:
                arr = self.data[key]
                if value != arr.chunks:
                    check_chunks_compatible(chunks=value, shape=arr.shape)
                    rechunk_recompress_array(self.data, key, chunks=value)

    def get_compressors(self) -> dict:
        assert self.backend == "zarr"
        compressors = dict()
        for key, value in self.data.items():
            compressors[key] = value.compressor
        return compressors

    def set_compressors(self, compressors: dict):
        assert self.backend == "zarr"
        for key, value in compressors.items():
            if key in self.data:
                arr = self.data[key]
                compressor = self.resolve_compressor(value)
                if compressor != arr.compressor:
                    rechunk_recompress_array(self.data, key, compressor=compressor)
