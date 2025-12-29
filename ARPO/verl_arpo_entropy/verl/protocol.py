# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
实现任意两个函数或模块之间的基础数据传输协议。
我们可以通过继承 Protocol 定义带有特定键的更详细的 batch 信息。
"""

import contextlib
import copy
import logging
import os
import pickle
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import ray
import tensordict
import torch
import torch.distributed
from packaging import version
from tensordict import TensorDict
from torch.utils.data import DataLoader

from verl.utils.device import get_torch_device
from verl.utils.py_functional import union_two_dict
from verl.utils.torch_functional import allgather_dict_tensors

__all__ = ["DataProto", "union_tensor_dict"]

# 如果 tensordict 支持，将其设置为非懒加载模式，防止兼容性异常
with contextlib.suppress(Exception):
    tensordict.set_lazy_legacy(False).set()


class _DataProtoConfigMeta(type):
    _config = {}

    # 自动填充的 key，用于元信息中判断是否支持自动 padding
    auto_padding_key = "_verl_auto_padding"

    @property
    def auto_padding(cls):
        # 优先检测环境变量 VERL_AUTO_PADDING，支持 TRUE/1 激活
        enabled_by_env = os.getenv("VERL_AUTO_PADDING", "FALSE").upper() in ["TRUE", "1"]
        return enabled_by_env or cls._config.get(cls.auto_padding_key, False)

    @auto_padding.setter
    def auto_padding(cls, enabled: bool):
        assert isinstance(enabled, bool), f"enabled 必须为 bool 类型，当前获得 {enabled} 类型为 {type(enabled)}"
        cls._config[cls.auto_padding_key] = enabled


class DataProtoConfig(metaclass=_DataProtoConfigMeta):
    pass

# padding size 相关特殊 key
_padding_size_key = "_padding_size_key_x123d"


def pad_dataproto_to_divisor(data: "DataProto", size_divisor: int):
    """将 DataProto pad 到 size_divisor 的倍数
    参数:
        size_divisor: 必须对 batch size 整除的分块数
    返回:
        data_padded: padding 后的 DataProto
        pad_size: 增补的数量
    """
    assert isinstance(data, DataProto), "data 必须为 DataProto 类型"
    if len(data) % size_divisor != 0:
        pad_size = size_divisor - len(data) % size_divisor
        padding_protos = []
        remaining_pad = pad_size
        while remaining_pad > 0:
            take_size = min(remaining_pad, len(data))
            padding_protos.append(data[:take_size])
            remaining_pad -= take_size
        data_padded = DataProto.concat([data] + padding_protos)
    else:
        if len(data) == 0:
            logging.warning("对空 DataProto 进行 padding，无需更改")
        pad_size = 0
        data_padded = data
    return data_padded, pad_size


def unpad_dataproto(data: "DataProto", pad_size):
    """将已 padding 的 DataProto 去除补上的部分"""
    if pad_size != 0:
        data = data[:-pad_size]
    return data


def union_tensor_dict(tensor_dict1: TensorDict, tensor_dict2: TensorDict) -> TensorDict:
    """合并（并集）两个 TensorDict，要求 batch size 一致。如果有相同 key，其张量全部内容必须一致。"""
    assert tensor_dict1.batch_size == tensor_dict2.batch_size, f"两个 tensor dict 的 batch size 必须一致，当前为 {tensor_dict1.batch_size} 和 {tensor_dict2.batch_size}"
    for key in tensor_dict2.keys():
        if key not in tensor_dict1.keys():
            tensor_dict1[key] = tensor_dict2[key]
        else:
            # 若有同名 key 强制检查值内容一致
            assert tensor_dict1[key].equal(tensor_dict2[key]), f"{key} 在 tensor_dict1 和 tensor_dict2 内容不同"
    return tensor_dict1


def union_numpy_dict(tensor_dict1: dict[str, np.ndarray], tensor_dict2: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """合并两个 numpy 字典，要求同名 key 的内容一致（允许 object 类型和 NaN），否则覆盖新增"""
    for key, val in tensor_dict2.items():
        if key in tensor_dict1:
            assert isinstance(tensor_dict2[key], np.ndarray)
            assert isinstance(tensor_dict1[key], np.ndarray)
            # 针对 nan/object 类型，用 pandas 验证一致性
            assert pd.DataFrame(tensor_dict2[key]).equals(pd.DataFrame(tensor_dict1[key])), f"{key} 在 tensor_dict1 和 tensor_dict2 内容不同"
        tensor_dict1[key] = val
    return tensor_dict1


def list_of_dict_to_dict_of_list(list_of_dict: list[dict]):
    """把字典的 list 转换为 key->value list 的字典"""
    if len(list_of_dict) == 0:
        return {}
    keys = list_of_dict[0].keys()
    output = {key: [] for key in keys}
    for data in list_of_dict:
        for key, item in data.items():
            assert key in output
            output[key].append(item)
    return output


def fold_batch_dim(data: "DataProto", new_batch_size):
    """
    折叠 batch 维度：形状从 [bsz, ...] 转为 [new_bsz, bsz//new_bsz, ...]
    """
    batch_size = data.batch.batch_size[0]
    assert batch_size % new_batch_size == 0

    tensor: TensorDict = data.batch
    non_tensor = data.non_tensor_batch

    tensor = tensor.view(new_batch_size, -1)
    tensor.auto_batch_size_(batch_dims=1)

    for key, val in non_tensor.items():
        non_tensor[key] = np.reshape(val, newshape=(new_batch_size, -1, *val.shape[1:]))

    return type(data)(batch=tensor, non_tensor_batch=non_tensor, meta_info=data.meta_info)


def unfold_batch_dim(data: "DataProto", batch_dims=2):
    """
    展开前 batch_dims 维，合并为新的 batch 维
    """
    tensor: TensorDict = data.batch
    non_tensor = data.non_tensor_batch
    tensor.auto_batch_size_(batch_dims=batch_dims)
    tensor = tensor.view(-1)

    batch_size = tensor.batch_size[0]

    non_tensor_new = {}

    for key, val in non_tensor.items():
        non_tensor_new[key] = np.reshape(val, newshape=(batch_size, *val.shape[batch_dims:]))

    return type(data)(batch=tensor, non_tensor_batch=non_tensor_new, meta_info=data.meta_info)


def collate_fn(x: list["DataProtoItem"]):
    """
    组装小 batch 数据为 DataProto
    """
    batch = []
    non_tensor_batch = []
    for data in x:
        batch.append(data.batch)
        non_tensor_batch.append(data.non_tensor_batch)
    batch = torch.stack(batch).contiguous()
    non_tensor_batch = list_of_dict_to_dict_of_list(non_tensor_batch)
    for key, val in non_tensor_batch.items():
        non_tensor_batch[key] = np.array(val, dtype=object)
    return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)


@dataclass
class DataProtoItem:
    # TODO(zhangchi.usc1992) 后续可加一致性校验
    batch: TensorDict = None
    non_tensor_batch: Dict = field(default_factory=dict)
    meta_info: Dict = field(default_factory=dict)


@dataclass
class DataProto:
    """
    DataProto 是用于标准化不同函数之间数据交换的数据协议。
    包含 batch（TensorDict 格式）和 meta_info（Dict 格式），batch 采用 TensorDict，方便高效处理和批量操作张量。
    理想情况下 batch 中的所有 tensor 都应该有相同的 batch size。
    """

    batch: TensorDict = None
    non_tensor_batch: Dict = field(default_factory=dict)
    meta_info: Dict = field(default_factory=dict)

    def __post_init__(self):
        # 初始化后做一次必要的结构检查
        self.check_consistency()

    def __len__(self):
        # 返回 DataProto 的 batch 对象长度
        if self.batch is not None:
            return self.batch.batch_size[0]
        elif self.non_tensor_batch is not None and len(self.non_tensor_batch) > 0:
            random_key = list(self.non_tensor_batch.keys())[0]
            return self.non_tensor_batch[random_key].shape[0]
        else:
            return 0

    def __getitem__(self, item):
        """
        支持多种索引方式的 DataProto 切片与单项访问。
        输入参数 item 可以为 int/slice/list/ndarray/tensor
        返回值：非整型返回 DataProto；整型返回 DataProtoItem
        """
        # 情形1: 切片对象（slice）调用自定义切片方法
        if isinstance(item, slice):
            return self.slice(item.start, item.stop, item.step)

        # 情形2: list、ndarray、tensor 则调用索引选择方法
        elif isinstance(item, (list, np.ndarray, torch.Tensor)):
            return self.select_idxs(item)

        # 情形3: 整数类，向后兼容返回 DataProtoItem
        elif isinstance(item, (int, np.integer)):
            tensor_data = self.batch[item] if self.batch is not None else None
            non_tensor_data = {key: val[item] for key, val in self.non_tensor_batch.items()}
            return DataProtoItem(batch=tensor_data, non_tensor_batch=non_tensor_data, meta_info=self.meta_info)

        # 不支持的索引类型
        else:
            raise TypeError(f"Indexing with {type(item)} is not supported")

    def __getstate__(self):
        import io

        buffer = io.BytesIO()
        # tensorDict 很新版要求连续内存和 consolidated
        if version.parse(tensordict.__version__) >= version.parse("0.5.0") and self.batch is not None:
            self.batch = self.batch.contiguous()
            self.batch = self.batch.consolidate()
        torch.save(self.batch, buffer)
        buffer_bytes = buffer.getvalue()
        return buffer_bytes, self.non_tensor_batch, self.meta_info

    def __setstate__(self, data):
        import io

        batch_deserialized_bytes, non_tensor_batch, meta_info = data
        # 反序列化张量，使用当前设备
        batch_deserialized = io.BytesIO(initial_bytes=batch_deserialized_bytes)
        batch = torch.load(batch_deserialized, weights_only=False, map_location="cpu" if not get_torch_device().is_available() else None)
        self.batch = batch
        self.non_tensor_batch = non_tensor_batch
        self.meta_info = meta_info

    def save_to_disk(self, filepath):
        """
        将 DataProto 持久化保存到磁盘文件
        """
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load_from_disk(filepath) -> "DataProto":
        """
        从磁盘文件载入 DataProto 数据
        """
        with open(filepath, "rb") as f:
            data = pickle.load(f)
            return data

    def print_size(self, prefix=""):
        """
        打印当前 DataProto 内 batch 及非 tensor_batch 各自占用的显存/内存量（GB）
        """
        size_of_tensordict = 0
        if self.batch is None:
            for key, tensor in self.batch.items():
                size_of_tensordict += tensor.element_size() * tensor.numel()
        size_of_numpy_array = 0
        for key, numpy_array in self.non_tensor_batch.items():
            size_of_numpy_array += numpy_array.nbytes

        size_of_numpy_array /= 1024**3
        size_of_tensordict /= 1024**3

        message = f"Size of tensordict: {size_of_tensordict} GB, size of non_tensor_batch: {size_of_numpy_array} GB"

        if prefix:
            message = f"{prefix}, " + message
        print(message)

    def check_consistency(self):
        """
        检查 DataProto 的 batch 和 non_tensor_batch 的一致性。
        主要保证 batch 为一维，non_tensor_batch 每个 key 其第0维长度等于 batch 大小。
        """
        if self.batch is not None:
            assert len(self.batch.batch_size) == 1, "只支持 num_batch_dims=1（即一维 batch）"

        if self.non_tensor_batch is not None:
            for key, val in self.non_tensor_batch.items():
                assert isinstance(val, np.ndarray)

        if self.batch is not None and self.non_tensor_batch is not None and len(self.non_tensor_batch) != 0:
            # 此限制如有需要可以去除（只限batch一维时允许非tensor字段）
            assert len(self.batch.batch_size) == 1, "当 non_tensor_batch 非空时只支持 batch 一维"

            batch_size = self.batch.batch_size[0]
            for key, val in self.non_tensor_batch.items():
                # print(f"key={key}, val={val}, type={type(val)}, shape={val.shape}, batch_size={batch_size}")
                assert isinstance(val, np.ndarray), f"non_tensor_batch 中数据必须是 numpy.array，且 dtype=object，key={key}, 类型={type(val)}"
                assert val.shape[0] == batch_size, f"key {key} 长度为 {len(val)}，却与 batch size {batch_size} 不符"

    @classmethod
    def from_single_dict(cls, data: Dict[str, Union[torch.Tensor, np.ndarray]], meta_info=None, auto_padding=False):
        """从 dict[{key: value}] 创建 DataProto，自动分辨 torch.Tensor 与 np.ndarray"""
        tensors = {}
        non_tensors = {}

        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                tensors[key] = val
            elif isinstance(val, np.ndarray):
                non_tensors[key] = val
            else:
                raise ValueError(f"不支持的数据类型 {type(val)}")

        return cls.from_dict(tensors=tensors, non_tensors=non_tensors, meta_info=meta_info, auto_padding=auto_padding)

    @classmethod
    def from_dict(cls, tensors: Optional[Dict[str, torch.Tensor]] = None, non_tensors=None, meta_info=None, num_batch_dims=1, auto_padding=False):
        """
        从 tensors/非 tensor 字典构建 DataProto。主要假设：
        1. tensors 中每个变量的 dim0 都相等
        2. 仅第0维作为 batch 维。
        """
        assert num_batch_dims > 0, "num_batch_dims 必须大于 0"
        if non_tensors is not None:
            assert num_batch_dims == 1, "当包含 non_tensors 时只支持 batch 维是 1"

        if tensors is None:
            tensors = {}
        if meta_info is None:
            meta_info = {}
        if non_tensors is None:
            non_tensors = {}

        assert isinstance(non_tensors, dict)

        # 检查并获取 batch size
        batch_size = None
        pivot_key = None
        for key, tensor in tensors.items():
            if batch_size is None:
                batch_size = tensor.shape[:num_batch_dims]
                pivot_key = key
            else:
                current_batch = tensor.shape[:num_batch_dims]
                assert batch_size == current_batch, f"tensors 所有变量 batch 维必须一致，pivot={pivot_key}={batch_size}，当前={key}={current_batch}"

        for key, val in non_tensors.items():
            if not isinstance(val, np.ndarray):
                non_tensors[key] = np.array(val, dtype=object)

        tensor_dict = TensorDict(source=tensors, batch_size=batch_size) if tensors else None
        if auto_padding:
            meta_info[DataProtoConfig.auto_padding_key] = True
        return cls(batch=tensor_dict, non_tensor_batch=non_tensors, meta_info=meta_info)

    def to(self, device) -> "DataProto":
        """
        将 batch 张量转移到指定设备
        参数:
            device: torch.device 或字符串
        返回:
            self: 转移设备后的 DataProto
        """
        if self.batch is not None:
            self.batch = self.batch.to(device)
        return self

    def select(self, batch_keys=None, non_tensor_batch_keys=None, meta_info_keys=None, deepcopy=False) -> "DataProto":
        """
        选择部分 batch/meta_info 字段，生成新 DataProto
        参数:
            batch_keys: 待选取的 batch 字段列表
            non_tensor_batch_keys: 待选取的非 tensor batch 字段
            meta_info_keys: 待选取的 meta_info 字段
            deepcopy: 是否做深拷贝
        返回:
            DataProto（选择后的数据）
        """
        # TODO (zhangchi.usc1992) 是否使用 copy
        if batch_keys is not None:
            batch_keys = tuple(batch_keys)
            sub_batch = self.batch.select(*batch_keys)
        else:
            sub_batch = self.batch

        if non_tensor_batch_keys is not None:
            non_tensor_batch = {key: val for key, val in self.non_tensor_batch.items() if key in non_tensor_batch_keys}
        else:
            non_tensor_batch = self.non_tensor_batch

        if deepcopy:
            non_tensor_batch = copy.deepcopy(non_tensor_batch)

        if meta_info_keys is not None:
            sub_meta_info = {key: val for key, val in self.meta_info.items() if key in meta_info_keys}
        else:
            sub_meta_info = self.meta_info

        if deepcopy:
            sub_meta_info = copy.deepcopy(sub_meta_info)

        return type(self)(batch=sub_batch, non_tensor_batch=non_tensor_batch, meta_info=sub_meta_info)

    def select_idxs(self, idxs):
        """
        按下标选取 DataProto 中的样本
        输入:
            idxs: torch.Tensor 或 np.ndarray 或 list，指定保留下标
        返回:
            DataProto: 新对象
        """
        if isinstance(idxs, list):
            idxs = torch.tensor(idxs)
            if idxs.dtype != torch.bool:
                idxs = idxs.type(torch.int32)

        if isinstance(idxs, np.ndarray):
            idxs_np = idxs
            idxs_torch = torch.from_numpy(idxs)
        else:  # torch.Tensor
            idxs_torch = idxs
            idxs_np = idxs.detach().cpu().numpy()

        batch_size = int(idxs_np.sum()) if idxs_np.dtype == bool else idxs_np.shape[0]

        if self.batch is not None:
            # TensorDict 原生支持下标
            selected_batch = TensorDict(source={key: tensor[idxs_torch] for key, tensor in self.batch.items()}, batch_size=(batch_size,), device=self.batch.device)
        else:
            selected_batch = None

        selected_non_tensor = {}
        for key, val in self.non_tensor_batch.items():
            selected_non_tensor[key] = val[idxs_np]

        return type(self)(batch=selected_batch, non_tensor_batch=selected_non_tensor, meta_info=self.meta_info)

    def slice(self, start=None, end=None, step=None):
        """
        切片并返回新的 DataProto（改良版，非单一元素，不直接返 DataProtoItem）
        参数:
            start/end/step: 切片起止，步长
        返回:
            DataProto（切片结果）
        """
        # 构造切片对象
        slice_obj = slice(start, end, step)

        # 处理 batch 张量
        if self.batch is not None:
            # TensorDict 支持原生切片
            sliced_batch = self.batch[slice_obj]
        else:
            sliced_batch = None

        # 处理非 tensor 字段
        sliced_non_tensor = {}
        for key, val in self.non_tensor_batch.items():
            sliced_non_tensor[key] = val[slice_obj]

        # 返回新 DataProto
        return type(self)(batch=sliced_batch, non_tensor_batch=sliced_non_tensor, meta_info=self.meta_info)

    def pop(self, batch_keys=None, non_tensor_batch_keys=None, meta_info_keys=None) -> "DataProto":
        """
        弹出（pop）指定 batch、non-tensor batch、meta_info 字段，作为新 DataProto 返回，并从自身移除
        参数:
            batch_keys, non_tensor_batch_keys, meta_info_keys: 需弹出的 key 列表
        返回:
            DataProto（弹出字段后的对象）
        """
        if batch_keys is None:
            batch_keys = []
        if meta_info_keys is None:
            meta_info_keys = []
        if non_tensor_batch_keys is None:
            non_tensor_batch_keys = []

        tensors = {}
        # 处理 tensor batch
        for key in batch_keys:
            assert key in self.batch.keys()
            tensors[key] = self.batch.pop(key)
        non_tensors = {}
        # 处理非 tensor batch
        for key in non_tensor_batch_keys:
            assert key in self.non_tensor_batch.keys()
            non_tensors[key] = self.non_tensor_batch.pop(key)
        meta_info = {}
        for key in meta_info_keys:
            assert key in self.meta_info.keys()
            meta_info[key] = self.meta_info.pop(key)
        return DataProto.from_dict(tensors=tensors, non_tensors=non_tensors, meta_info=meta_info)

    def rename(self, old_keys=None, new_keys=None) -> "DataProto":
        """
        重命名 batch 里的 key，不改变其它部分
        参数:
            old_keys, new_keys: 旧键/新键列表（等长）
        返回:
            本身
        """

        def validate_input(keys):
            if keys is not None:
                if isinstance(keys, str):
                    keys = [keys]
                elif isinstance(keys, list):
                    pass
                else:
                    raise TypeError(f"keys 必须为 list 或 str，但得到 {type(keys)}")
            return keys

        old_keys = validate_input(old_keys)
        new_keys = validate_input(new_keys)

        if len(new_keys) != len(old_keys):
            raise ValueError(f"new_keys 与 old_keys 长度必须一致，当前分别为 {len(new_keys)} 和 {len(old_keys)}")

        self.batch.rename_key_(tuple(old_keys), tuple(new_keys))

        return self

    def union(self, other: "DataProto") -> "DataProto":
        """
        与其它 DataProto 合并 batch、non-tensor batch 和 meta_info。冲突 key 内容必须一致，否则报错。
        参数:
            other: 另一个待合并的 DataProto
        返回:
            self（合并后的结果）
        """
        self.batch = union_tensor_dict(self.batch, other.batch)
        self.non_tensor_batch = union_numpy_dict(self.non_tensor_batch, other.non_tensor_batch)
        self.meta_info = union_two_dict(self.meta_info, other.meta_info)
        return self

    def make_iterator(self, mini_batch_size, epochs, seed=None, dataloader_kwargs=None):
        r"""
        构造一个 mini-batch 迭代器，用于 batch 训练循环。
        要保证总 batch 大小能被 mini_batch_size 整除。
        seed 可以设定迭代种子，实现可重复性。
        返回:
            生成器，每次返回一个 mini-batch DataProto。
        """
        assert self.batch.batch_size[0] % mini_batch_size == 0, f"{self.batch.batch_size[0]} % {mini_batch_size} != 0"
        # 可直接用 pytorch DataLoader iter TensorDict
        if dataloader_kwargs is None:
            dataloader_kwargs = {}

        if seed is not None:
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = None

        assert isinstance(dataloader_kwargs, Dict)
        train_dataloader = DataLoader(dataset=self, batch_size=mini_batch_size, collate_fn=collate_fn, generator=generator, **dataloader_kwargs)

        def get_data():
            for _ in range(epochs):
                for d in train_dataloader:
                    d.meta_info = self.meta_info
                    yield d

        return iter(get_data())

    def is_padding_enabled(self):
        """
        判断当前 DataProto 是否允许自动 padding。
        返回:
            是否启用 padding（bool）
        """
        dataproto_specific_padding = self.meta_info.get(DataProtoConfig.auto_padding_key, False)
        return dataproto_specific_padding or DataProtoConfig.auto_padding

    def padding(self, padding_size, padding_candidate=""):
        """
        通过将指定 padding_candidate（"first"/"last"）重复 padding_size 次并拼接至当前 DataProto，实现 pad
        参数:
            padding_size: padding 行数
            padding_candidate: 采用“first”或“last”行作为 padding
        """
        if padding_size == 0:
            return
        padding_candidate = self.select_idxs([0 if padding_candidate == "first" else len(self) - 1])
        padding_part = padding_candidate.repeat(padding_size)
        padded_dp = DataProto.concat([self, padding_part])
        self.batch = padded_dp.batch
        self.non_tensor_batch = padded_dp.non_tensor_batch

    def chunk(self, chunks: int) -> List["DataProto"]:
        """
        将 DataProto 按 batch 维等份分 chunk，要求为等分（如没启用 padding）。
        每个分块继承 meta_info。
        参数:
            chunks: chunk 数
        返回:
            List[DataProto]: 拆分结果
        """
        if not self.is_padding_enabled():
            assert len(self) % chunks == 0, f"只支持等分，DataProto size={len(self)}，chunk={chunks}"

        bsz_in_batch = None
        if self.batch is not None:
            batch_lst = self.batch.chunk(chunks=chunks, dim=0)
            bsz_in_batch = np.array([batch.batch_size[0] for batch in batch_lst])
            chunk_indices = np.cumsum(bsz_in_batch)[:-1]
        else:
            batch_lst = [None for _ in range(chunks)]

        non_tensor_batch_lst = [{} for _ in range(chunks)]
        for key, val in self.non_tensor_batch.items():
            assert isinstance(val, np.ndarray)
            if bsz_in_batch is not None:
                non_tensor_lst = np.array_split(val, chunk_indices.tolist())
            else:
                non_tensor_lst = np.array_split(val, chunks)
            assert len(non_tensor_lst) == chunks
            for i in range(chunks):
                non_tensor_batch_lst[i][key] = non_tensor_lst[i]

        output = []
        for i in range(chunks):
            output.append(type(self)(batch=batch_lst[i], non_tensor_batch=non_tensor_batch_lst[i], meta_info=self.meta_info))

        return output

    @staticmethod
    def concat(data: List["DataProto"]) -> "DataProto":
        """
        合并一组 DataProto（沿 batch 第0维拼接），meta_info 假定一致取第一个
        参数:
            data: DataProto 列表
        返回:
            合并后的 DataProto
        """
        batch_lst = []
        for batch in data:
            batch_lst.append(batch.batch)
        new_batch = torch.cat(batch_lst, dim=0) if batch_lst[0] is not None else None

        non_tensor_batch = list_of_dict_to_dict_of_list(list_of_dict=[d.non_tensor_batch for d in data])
        for key, val in non_tensor_batch.items():
            non_tensor_batch[key] = np.concatenate(val, axis=0)

        cls = type(data[0]) if len(data) > 0 else DataProto
        return cls(batch=new_batch, non_tensor_batch=non_tensor_batch, meta_info=data[0].meta_info)

    def reorder(self, indices):
        """
        对 DataProto 进行批内样本重排序（原地操作）
        """
        indices_np = indices.detach().numpy()
        self.batch = self.batch[indices]
        self.non_tensor_batch = {key: val[indices_np] for key, val in self.non_tensor_batch.items()}

    def repeat(self, repeat_times=2, interleave=True):
        """
        批量整体性/顺序重复 DataProto。repeat_times 为每条重复次数。
        interleave 控制是顺序堆叠还是交错
        返回：
            新 DataProto（重复数据）
        """
        if self.batch is not None:
            if interleave:
                # 交错重复
                repeated_tensors = {key: tensor.repeat_interleave(repeat_times, dim=0) for key, tensor in self.batch.items()}
            else:
                # 批次堆叠
                repeated_tensors = {key: tensor.unsqueeze(0).expand(repeat_times, *tensor.shape).reshape(-1, *tensor.shape[1:]) for key, tensor in self.batch.items()}

            repeated_batch = TensorDict(
                source=repeated_tensors,
                batch_size=(self.batch.batch_size[0] * repeat_times,),
            )
        else:
            repeated_batch = None

        repeated_non_tensor_batch = {}
        for key, val in self.non_tensor_batch.items():
            if interleave:
                repeated_non_tensor_batch[key] = np.repeat(val, repeat_times, axis=0)
            else:
                repeated_non_tensor_batch[key] = np.tile(val, (repeat_times,) + (1,) * (val.ndim - 1))

        return type(self)(
            batch=repeated_batch,
            non_tensor_batch=repeated_non_tensor_batch,
            meta_info=self.meta_info,
        )

    def unfold_column_chunks(self, n_split: int, split_keys: Optional[List[str]] = None):
        """
        将二维张量 (如[bsz, n, ...])，把 n 拆下来拼为 batch 维，适合保存组内不混洗数据。
        指定 split_keys 时只拆对应字段，未指定时默认所有键都处理。
        """
        if self.batch is not None:
            unfolded_batch = {}
            for key in self.batch.keys():
                if key in split_keys if split_keys is not None else False:
                    shape = list(self.batch[key].shape)
                    shape[0] = self.batch[key].shape[0] * n_split
                    shape[1] = self.batch[key].shape[1] // n_split
                    unfolded_batch[key] = self.batch[key].reshape(*shape)
                else:
                    unfolded_batch[key] = torch.repeat_interleave(self.batch[key], n_split, dim=0)
            # 将 unfolded_batch 作为 TensorDict，放回原 batch 的设备上
            unfolded_batch = TensorDict(source=unfolded_batch, batch_size=(self.batch.batch_size[0] * n_split,), device=self.batch.device)
        else:
            unfolded_batch = None

        repeated_non_tensor_batch = {}
        for key, val in self.non_tensor_batch.items():
            if key in split_keys:
                shape = list(val.shape)
                shape[0] = val.shape[0] * n_split
                shape[1] = val.shape[1] // n_split
                repeated_non_tensor_batch[key] = val.reshape(*shape)
            else:
                repeated_non_tensor_batch[key] = np.repeat(val, n_split, axis=0)

        return type(self)(
            batch=unfolded_batch,
            non_tensor_batch=repeated_non_tensor_batch,
            meta_info=self.meta_info,
        )

    def sample_level_repeat(self, repeat_times):
        """
        按样本级别指定 repeat_times 多次重复每一行（支持不规则的重复数）。
        参数：
            repeat_times: 每一条样本的重复次数，支持 list、tensor、ndarray、tuple
        返回：
            新 DataProto
        """
        if isinstance(repeat_times, tuple):
            repeat_times = list(repeat_times)
        elif isinstance(repeat_times, torch.Tensor):
            assert len(repeat_times.shape) == 1
            repeat_times = repeat_times.tolist()
        elif isinstance(repeat_times, np.ndarray):
            assert len(repeat_times.shape) == 1
            repeat_times = repeat_times.tolist()
        else:
            assert isinstance(repeat_times, list), f"repeat_times 必须为 list、torch.Tensor、np.ndarray 或 tuple，当前为 {type(repeat_times)}"
        repeat_times = torch.tensor(repeat_times)

        if self.batch is not None:
            # 交错式 repeat
            repeated_tensors = {key: tensor.repeat_interleave(repeat_times, dim=0) for key, tensor in self.batch.items()}

            repeated_batch = TensorDict(
                source=repeated_tensors,
                batch_size=(repeat_times.sum().item(),),
                device=self.batch.device,
            )
        else:
            repeated_batch = None

        repeated_non_tensor_batch = {}
        for key, val in self.non_tensor_batch.items():
            repeated_non_tensor_batch[key] = np.repeat(val, repeat_times, axis=0)

        return type(self)(
            batch=repeated_batch,
            non_tensor_batch=repeated_non_tensor_batch,
            meta_info=self.meta_info,
        )


@dataclass
class DataProtoFuture:
    """
    DataProtoFuture 设计用于在 driver 端消除数据获取阻塞，实现异步执行。
    - collect_fn 是一个 Callable，将一组 ray future 聚合为 DataProto。
    - dispatch_fn 可将 DataProto 拆分为 world_size 大小的多份。
    注意：DataProtoFuture 只支持从函数/方法产出的 future 直接传递为下个输入，driver 端不能对其做其它操作。
    """

    collect_fn: Callable
    futures: List[ray.ObjectRef]
    dispatch_fn: Callable = None

    @staticmethod
    def concat(data: List[ray.ObjectRef]) -> "DataProtoFuture":
        output = DataProtoFuture(collect_fn=DataProto.concat, futures=data)
        return output

    def chunk(self, chunks: int) -> List["DataProtoFuture"]:
        """
        将 future 列表拆分为 chunks 份封装为新 DataProtoFuture
        """
        from functools import partial

        arg_future_lst = []
        for i in range(chunks):
            # 注意 partial dispatch_fn 参数传递
            def dispatch_fn(x, i, chunks):
                return x.chunk(chunks=chunks)[i]

            arg_future = DataProtoFuture(collect_fn=self.collect_fn, dispatch_fn=partial(dispatch_fn, i=i, chunks=chunks), futures=self.futures)
            arg_future_lst.append(arg_future)
        return arg_future_lst

    def get(self):
        """
        获取实际计算结果，将一组 future 聚合为 DataProto，并根据需要做 dispatch 拆分
        """
        output = ray.get(self.futures)  # dp_size.
        for o in output:
            assert isinstance(o, DataProto)
        output = self.collect_fn(output)  # select dp, concat
        if self.dispatch_fn is not None:
            output = self.dispatch_fn(output)  # batch 维度 split（如 chunk 场景）
        return output


def all_gather_data_proto(data: DataProto, process_group):
    """
    DataProto 的多进程 all_gather 操作（原地修改对象）
    包括 tensor batch 及非 tensor batch 两部分的聚合
    """
    # 获取进程组内世界大小
    group_size = torch.distributed.get_world_size(group=process_group)
    assert isinstance(data, DataProto)
    prev_device = data.batch.device
    # 先转到当前可用设备
    data.batch = data.batch.to(get_torch_device().current_device())
    # 聚合 batch（TensorDict）
    data.batch = allgather_dict_tensors(data.batch.contiguous(), size=group_size, group=process_group, dim=0)
    data.batch = data.batch.to(prev_device)
    # all gather 非 tensor_batch
    all_non_tensor_batch = [None for _ in range(group_size)]
    torch.distributed.all_gather_object(all_non_tensor_batch, data.non_tensor_batch, group=process_group)
    data.non_tensor_batch = {k: np.concatenate([d[k] for d in all_non_tensor_batch]) for k in data.non_tensor_batch}
