import re
from copy import deepcopy
from findspingroup.version import __version__
import numpy as np

from spglib import SpglibDataset
from functools import cached_property

from findspingroup.structure.cell import AtomicSite
from findspingroup.core.identify_symmetry_from_ops import deduplicate_matrix_pairs, get_space_group_from_operations, \
    get_arithmetic_crystal_class_from_ops, identify_point_group, get_magnetic_space_group_from_operations
from findspingroup.utils.matrix_utils import getNormInf, integerize_matrix, rref_with_tolerance, in_space_group, \
    normalize_vector_to_zero


def parse_label_and_value(text):
    label, value = text.split(':', 1)
    return label, value



class BrillouinZoneMatcher:
    def __init__(self, rules):
        self.parsed_rules = []
        for label, pattern, splitting in rules:
            parsed = self._parse_pattern(pattern)
            score = self._calculate_specificity_score(parsed)
            self.parsed_rules.append({
                'label': label,
                'pattern': parsed,
                'splitting': splitting,
                'score': score
            })

        self.parsed_rules.sort(key=lambda x: x['score'], reverse=True)

    def _parse_pattern(self, pattern_str):
        content = pattern_str.strip("()").split(",")
        parsed = []
        for item in content:
            item = item.strip()

            try:
                val = float(eval(item, {"__builtins__": None}, {}))
                parsed.append({'type': 'fixed', 'val': val})
            except:
                parsed.append({'type': 'var', 'name': item})
        return parsed

    def _calculate_specificity_score(self, parsed_pattern):
        score = 0
        vars_seen = set()

        for p in parsed_pattern:
            if p['type'] == 'fixed':
                score += 10
            elif p['type'] == 'var':
                if p['name'] in vars_seen:
                    score += 5
                vars_seen.add(p['name'])
        return score

    def check(self, u, v, w, tol=1e-5):
        input_k = np.mod([u, v, w],1)

        for rule in self.parsed_rules:
            match = True
            var_map = {}

            for i in range(3):
                rule_comp = rule['pattern'][i]
                input_val = input_k[i]

                if rule_comp['type'] == 'fixed':
                    if abs(input_val - rule_comp['val']) > tol:
                        match = False
                        break

                elif rule_comp['type'] == 'var':
                    var_name = rule_comp['name']
                    if var_name in var_map:
                        if abs(input_val - var_map[var_name]) > tol:
                            match = False
                            break
                    else:
                        var_map[var_name] = input_val

            if match:
                return {
                    "matched_label": rule['label'],
                    "has_splitting": rule['splitting'],
                    "k_point": (u, v, w)
                }

        raise ValueError(f"No matching rule found for k-point ({u}, {v}, {w})")


def write_kpoints(seekpath_out, matcher: BrillouinZoneMatcher, num_points=40, extra_kpoints=None):
    """
    Write k-point path string with SOC splitting info for Endpoints AND Path.
    """
    kpts = seekpath_out['point_coords']
    path = seekpath_out['path']

    def append_low_sym_points_simple_chain(extra_points):
        """
        将所有额外点按输入顺序连成一条链，首尾连接 Gamma 点。
        路径逻辑: GAMMA -> P1 -> P2 -> ... -> Pn -> GAMMA

        参数:
        extra_points: list of tuples, e.g. [([0.1, 0, 0], "MyP1"), ([0.2, 0, 0], "MyP2")]
        seekpath_output: dict, seekpath.get_path() 的输出结果
        """

        # 1. 复制基础数据
        point_coords = {}
        path_list = []

        # 如果没有额外点，直接返回
        if not extra_points:
            return {'point_coords': point_coords, 'path': path_list}

        # 2. 将所有新坐标加入字典，并按顺序记录 Label
        new_labels_ordered = []
        for coords, label in extra_points:
            point_coords[label] = np.array(coords)
            new_labels_ordered.append(label)

        # 3. 构建链式路径
        # -------------------------------------------------

        # 确定 Gamma 点的标签 (Seekpath 默认通常是 'GAMMA')
        # 如果你的系统里叫 'GAM' 或其他名字，请在这里修改
        gamma_label = 'GAMMA'

        # A. 起始段：GAMMA -> 第一个额外点
        first_point = new_labels_ordered[0]
        path_list.append((gamma_label, first_point))

        # B. 中间段：点1 -> 点2 -> ... -> 点N
        # 按照你在列表里提供的顺序连接
        for i in range(len(new_labels_ordered) - 1):
            current_p = new_labels_ordered[i]
            next_p = new_labels_ordered[i + 1]
            path_list.append((current_p, next_p))

        # C. 结束段：最后一个额外点 -> GAMMA
        last_point = new_labels_ordered[-1]
        path_list.append((last_point, gamma_label))

        return {'point_coords': point_coords, 'path': path_list}

    def fmt(label):
        # 处理特殊字符，如把 GAMMA 转为 Γ
        return 'Γ' if label == 'GAMMA' else label.replace('_', '')

    # --- 内部辅助函数：获取某一个 k 点的劈裂状态 ---
    def get_split_status(u, v, w):
        """
        输入 k 点坐标，返回 (matched_label, is_splitting)
        """
        result = matcher.check(u, v, w)
        if result:
            return result['matched_label'], result['has_splitting']
        else:
            # 如果没匹配到（通常不应该发生在高对称点），采用保守策略
            return "Unknown", True

    # --- 内部辅助函数：生成显示的 Tag 字符串 ---
    def make_tag(label, is_splitting, is_path=False):
        """
        格式化输出字符串。
        is_path=True 时用于显示路径信息，False 用于显示端点信息
        """

        # 如果是劈裂，加上 *** 高亮
        highlight = "***" if is_splitting else ""

        if is_path:
            # 路径显示的格式： | Label : Status
            return f"| {label} {highlight}"
        else:
            # 点显示的格式： {Label: Status}
            return f"{highlight}"

    def _write_kpoints(s_head,path_list,kpts_list):
        path_label = []
        for start_label, end_label in path_list:
            # 1. 获取坐标
            k1 = np.array(kpts_list[start_label])
            k2 = np.array(kpts_list[end_label])

            # 2. 计算中点 (代表路径)
            mid_k = (k1 + k2) / 2.0

            # 3. 分别获取 起点、终点、路径 的状态
            #    注意：matcher.check 可能会返回具体的 Little Group 名字
            lbl_start, split_start = get_split_status(k1[0], k1[1], k1[2])
            lbl_end, split_end = get_split_status(k2[0], k2[1], k2[2])
            lbl_mid, split_mid = get_split_status(mid_k[0], mid_k[1], mid_k[2])

            path_label.append(lbl_mid)

            # 4. 生成对应的文本 Tag
            tag_start_pt = make_tag(lbl_start, split_start, is_path=False)
            tag_end_pt = make_tag(lbl_end, split_end, is_path=False)
            tag_path = make_tag(lbl_mid, split_mid, is_path=True)

            # 5. 写入起点行
            #    格式: 坐标 ! 原始Label {实际对称性Label: 劈裂情况}
            s_head.append(
                f"{k1[0]:10.6f} {k1[1]:10.6f} {k1[2]:10.6f} ! "
                f"{fmt(start_label) + ' ' + tag_start_pt:<9}\n"
            )

            # 6. 写入终点行
            #    格式: 坐标 ! 原始Label {实际对称性Label: 劈裂情况} | Path[路径Label]: 劈裂情况
            #    我们将路径的信息附着在终点行后面，这样看起来像是： Start -> End (via Path)
            s_head.append(
                f"{k2[0]:10.6f} {k2[1]:10.6f} {k2[2]:10.6f} ! "
                f"{fmt(end_label) + ' ' + tag_end_pt:<9} {tag_path}\n"
            )

            # 7. 分隔空行
            s_head.append("\n")
        return path_label,s_head

    s = []
    # 写入文件头
    s.append(f"Generated by seekpath and findspingroup v{__version__} (*** for spin splitting)\n ")
    s.append(f"{num_points}\nLine-mode\nReciprocal\n")
    path_label,s = _write_kpoints(s,path,kpts)

    if extra_kpoints:
        add_kpoints = []
        for i in extra_kpoints:
            if matcher.check(*i[0])['matched_label'] in path_label:
                continue
            else:
                add_kpoints.append(i)
        extra_seekpath = append_low_sym_points_simple_chain(add_kpoints)
        s = _write_kpoints(s,extra_seekpath['path'],extra_seekpath['point_coords']|kpts)[1]
    return ''.join(s)

def find_uvw_whole_string(data_list):
    """
    检查整个字符串是否包含 'u', 'v', 'w' 中的任意两个及以上。
    如果符合，返回 (冒号前的字符串, 索引)。

    Args:
        data_list (list): 字符串列表

    Returns:
        list: [Label]
    """
    target_chars = {'u', 'v', 'w'}
    indices = []
    for index, text in enumerate(data_list):
        # 1. 核心变化：对【整个字符串 text】进行集合运算判断
        common_chars = set(text) & target_chars

        if len(common_chars) >= 2:
            # 2. 提取冒号前的部分用于返回
            indices.append(index)


    return indices

def op_key(op):
    rot1, rot2, t = op   # 每个元素是 [rot1, rot2, t]
    rot1 = np.asarray(rot1)
    rot2 = np.asarray(rot2)
    t    = np.asarray(t)

    # 用 Frobenius 范数衡量“距离单位矩阵有多远”
    d_rot2 = np.linalg.norm(rot2 - np.identity(3), ord='fro')
    d_t    = np.linalg.norm(t)           # t 与 [0,0,0] 的距离
    d_rot1 = np.linalg.norm(rot1 - np.identity(3), ord='fro')

    # 返回一个“排序用的三元组”：先比 rot2，再比 t，再比 rot1
    return (d_rot2, d_t, d_rot1)

def check_divisible(a, b):
    if b == 0:
        raise ValueError("cannot divide by zero")
    if a % b != 0:
        raise ValueError(f"{a} is not divisible by {b}")
    return a // b  # 返回整除结果


def _validate_array_format(array, expected_shape):
    """Unified validation function for arrays with expected shapes."""
    array = np.array(array, dtype=np.float64)

    # Handle translation vector special case (accept both (3,) and (3,1))
    if array.shape == (3,1):
        array = array.reshape(3, )

    if array.shape != expected_shape:
        raise ValueError(f"must have shape {expected_shape}, got shape {array.shape}")

    return array

def find_group_generators(ops: list) -> list:
    """Find a minimal set of generators for the group defined by ops."""
    generators = []
    current_group = set()
    # todo: this is not finished yet.
    for op in ops:
        if op not in current_group:
            generators.append(op)
            # Generate new elements by combining with existing group elements
            new_elements = set()
            for g in current_group:
                new_elements.add(g @ op)
                new_elements.add(op @ g)
            new_elements.add(op)
            current_group.update(new_elements)

    return generators

def integer_points_in_new_cell(T, tol=1e-5):
    """
    输入:
        T: 3x3 矩阵，行向量是新的基矢在旧基下的表示 row vectors
    输出:
        所有落在由这三个基矢张成的 unit cell 内的整数点 (i, j, k) 列表
    """
    T = np.asarray(T, dtype=float)

    # 1) 找到 unit cell 的 8 个顶点：u in {0,1}^3
    corners = np.array([[i, j, k]
                        for i in [0, 1]
                        for j in [0, 1]
                        for k in [0, 1]], dtype=float)
    vertices = corners @ T              # shape: (8, 3)
    # print(vertices)
    # 2) 轴向包围盒
    mins = np.floor(vertices.min(axis=0)).astype(int)
    maxs = np.ceil(vertices.max(axis=0)).astype(int)

    # 3) 预先算好 T 的逆，用来从整数点算回 u
    invT = np.linalg.inv(T)

    points = []
    for i in range(mins[0], maxs[0] + 1):
        for j in range(mins[1], maxs[1] + 1):
            for k in range(mins[2], maxs[2] + 1):
                n = np.array([i, j, k], dtype=float)  # 整数点
                u = n @ invT                          # 求对应的 u
                # print('n',n,'u',u)
                # 4) 判断 u 是否在 [0,1)^3（加一点浮点误差容忍）
                if np.all(u >= -tol) and np.all(u < 1 - tol):
                    points.append((i, j, k))

    return points


class SpinSpaceGroupOperation:
    """
    Represents a spin space group operation consisting of a rotation, translation (mod 1), and spin rotation.

    Attributes:
        rotation (np.ndarray): A 3x3 rotation matrix.
        translation (np.ndarray): A 3x1 translation vector.
        spin_rotation (np.ndarray): A 3x3 spin rotation matrix.

    Methods:
        __matmul__(other): Composes two symmetry operations or acts on an atomic site or vector.
        inv(): Returns the inverse of the spin space group operation.
        to_spg_op(): Converts to SpinPointGroupOperation by dropping the translation part.

        tolist(): Returns the operation as a list [spin_rotation, rotation, translation].


    """
    def __init__(self, spin_rotation, rotation, translation ):
        self.rotation = _validate_array_format(rotation,(3,3))  # 3x3 matrix
        self.translation = _validate_array_format(translation,(3,))  # 3x1 vector
        self.spin_rotation = _validate_array_format(spin_rotation,(3,3))  # 3x3 matrix
        self._data = [self.spin_rotation, self.rotation, self.translation ]


    def __getitem__(self, index):
        return self._data[index]

    def __len__(self):
        return len(self._data)

    def __repr__(self):
        return f"SpinOperation(spin_rotation={self.spin_rotation}|| rotation={self.rotation}| translation={self.translation})"

    def __matmul__(self, other):
        if isinstance(other, SpinSpaceGroupOperation):
            # compose two symmetry operations
            new_rotation = self.rotation @ other.rotation
            new_translation = normalize_vector_to_zero( self.rotation @ other.translation + self.translation,atol=1e-4)
            new_spin_rotation = self.spin_rotation @ other.spin_rotation
            # constructor expects (spin_rotation, rotation, translation)
            return SpinSpaceGroupOperation(new_spin_rotation, new_rotation, new_translation)

        elif isinstance(other, AtomicSite):
            # act on an atomic site
            new_position = normalize_vector_to_zero(self.rotation @ other.position + self.translation,atol=1e-9)
            new_magnetic_moment = self.spin_rotation @ other.magnetic_moment
            return AtomicSite(new_position, new_magnetic_moment,other.occupancy, other.element_symbol)

        elif isinstance(other, np.ndarray):
            # act on normal vector or [spin, position] vector
            if other.shape == (3,):
                new_vector = self.rotation @ other  # only rotation
                return new_vector
            elif other.shape == (6,):
                position = self.rotation @ other [3:6] + self.translation.flatten()
                magnetic_moment = self.spin_rotation @ other[0:3]
                return position + magnetic_moment
            else:
                raise ValueError("Unsupported ndarray shape for SpinOperation @ vector")

        else:
            raise TypeError("SpinOperation @ unsupported type")

    def inv(self) -> 'SpinSpaceGroupOperation':
        """Inverse of the spin space group operation. {Rs||R|t} -> {Rs^{-1}||R^{-1}|-R{-1}*t}"""
        inv_rotation = np.linalg.inv(self.rotation)
        inv_translation = normalize_vector_to_zero(-inv_rotation @ self.translation ,atol=1e-4)
        inv_spin_rotation = np.linalg.inv(self.spin_rotation)
        return SpinSpaceGroupOperation(inv_spin_rotation, inv_rotation, inv_translation)

    def to_spg_op(self) -> 'SpinPointGroupOperation':
        """Convert to SpinPointGroupOperation by dropping the translation part."""
        return SpinPointGroupOperation(deepcopy(self.spin_rotation), deepcopy(self.rotation))

    def tolist(self):
        return [self.spin_rotation.round(6).tolist(), self.rotation.round(6).tolist(), self.translation.round(6).tolist()]

    @classmethod
    def identity(cls) -> 'SpinSpaceGroupOperation':
        """Returns the identity operation."""
        return cls(np.eye(3), np.eye(3), np.zeros(3))

    def is_same_with(self,other, atol=1e-3):
        A1, B1, C1 = self._data
        A2, B2, C2 = other._data
        if np.allclose(A1, A2, atol=atol) and np.allclose(B1, B2, atol=atol) and getNormInf(C1, C2) < atol:
            return True
        else:
            return False

class SpinPointGroupOperation:
    """
    Represents a spin point group operation consisting of a rotation and spin rotation.

    Attributes:
        rotation (np.ndarray): A 3x3 rotation matrix.
        spin_rotation (np.ndarray): A 3x3 spin rotation matrix.

    Methods:
        __matmul__(other): Composes two symmetry operations or acts on an atomic site or vector.
        inv(): Returns the inverse of the spin point group operation.
        act_on_kpoint(k_point): Acts on a k-point in reciprocal space.
        effective_k_operation(): Returns the effective k-point operation.

        tolist(): Returns the operation as a list [spin_rotation, rotation].
    """
    def __init__(self,spin_rotation, rotation):
        self.rotation = _validate_array_format(rotation,(3,3))  # 3x3 matrix
        self.spin_rotation = _validate_array_format(spin_rotation,(3,3))  # 3x3 matrix


        self._data = [ self.spin_rotation, self.rotation]
        self._spin_det_sign = np.sign(np.linalg.det(self.spin_rotation))

    def __getitem__(self, index):
        return self._data[index]

    def __len__(self):
        return len(self._data)

    def __repr__(self):
        return f"SpinPointGroupOperation(spin_rotation={self.spin_rotation}|| rotation={self.rotation})"


    def __matmul__(self, other):
        if isinstance(other, SpinPointGroupOperation):
            # compose two symmetry operations
            new_rotation = self.rotation @ other.rotation
            new_spin_rotation = self.spin_rotation @ other.spin_rotation
            # constructor expects (spin_rotation, rotation)
            return SpinPointGroupOperation(new_spin_rotation, new_rotation)

        elif isinstance(other, AtomicSite):
            # act on an atomic site
            new_position = normalize_vector_to_zero(self.rotation @ other.position )
            new_magnetic_moment = self.spin_rotation @ other.magnetic_moment
            return AtomicSite(new_position, new_magnetic_moment,other.occupancy, other.element_symbol)

        elif isinstance(other, np.ndarray):
            # act on normal vector or [spin, position] vector
            if other.shape == (3,):
                new_vector = self.rotation @ other  # only rotation
                return new_vector
            elif other.shape == (6,):
                position = self.rotation @ other [3:6]
                magnetic_moment = self.spin_rotation @ other[0:3]
                return position + magnetic_moment
            else:
                raise ValueError("Unsupported ndarray shape for SpinPointGroupOperation @ vector")

        else:
            raise TypeError("SpinPointGroupOperation @ unsupported type")

    def inv(self):
        """Inverse of the spin point group operation. {Rs||R} -> {Rs^{-1}||R^{-1}}"""
        inv_rotation = np.linalg.inv(self.rotation)
        inv_spin_rotation = np.linalg.inv(self.spin_rotation)
        return SpinPointGroupOperation(inv_spin_rotation, inv_rotation)

    def act_on_kpoint(self, k_point):
        """Acts on a k-point in reciprocal space."""
        k_point = _validate_array_format(k_point, (3,))
        new_k_point = self.rotation @ k_point * self._spin_det_sign % 1
        return new_k_point

    def effective_k_operation(self):
        """Returns the effective k-point operation."""
        return self.rotation * self._spin_det_sign

    def tolist(self):
        return [self.spin_rotation, self.rotation]


def fetch_ssg_by_index(index:str):
    """Fetch the spin space group operations from database by its index.
    This is a placeholder function. In a real implementation, this would query a database or a predefined dictionary.
    """



    # Example placeholder data
    SSG_DATA = {}
    example_generators = SSG_DATA.get(index, False)

    if not example_generators:
        raise ValueError(f"Spin space group with index {index} not found in the database.")
    else:
        return example_generators





# 假设必要的外部库和函数已经导入 (fsg.data, op_key, identify_point_group 等)

class SpinSpaceGroup:
    """
    Represents a spin space group defined by its symmetry operations.
    Refactored for lazy evaluation using cached_property.
    """

    def __init__(self, input_data: str | list):
        """
        Initializes a SpinSpaceGroup instance.
        """
        self.tol = 0.03
        self._input_index = None  # 用于存储字符串输入时的 index
        self._input_ops = []  # 用于存储初始传入的操作列表

        # --- 1. 解析输入 (保留原有逻辑) ---
        if isinstance(input_data, str):
            try:
                self._input_ops = fetch_ssg_by_index(input_data)
            except ValueError:
                raise ValueError(f"Spin space group with index {input_data} not found in the database.")

            self._input_index = input_data
            self.lattice_settings = 'G0_standard'
            self.spin_settings = 'cartesian'
            self.relative_settings = 'Arbitrary'

        elif isinstance(input_data, list):
            if all(isinstance(op, SpinSpaceGroupOperation) for op in input_data):
                self._input_ops = input_data
            elif all(isinstance(op, list) and len(op) == 3 for op in input_data):
                self._input_ops = [SpinSpaceGroupOperation(op[0], op[1], op[2]) for op in input_data]
            else:
                raise ValueError(
                    "List must contain either SpinOperation instances or lists of [rotation, translation, spin_rotation].")

            self.settings = 'primitive'
            self.spin_settings = 'lattice'
            self.relative_settings = 'OSSG'

        else:
            raise TypeError("Input must be either a string index or a list of SpinOperation instances or lists.")

    # =========================================================================
    # 核心属性 (Lazy Loading)
    # =========================================================================

    @cached_property
    def ops(self):
        """返回排序后的操作列表"""
        # 对应原 _analyze_structure 第一行: sorted(self.ops, key=op_key)
        return sorted(self._input_ops, key=op_key)

    @cached_property
    def spin_translation_group(self):
        return self.get_spin_translation_group()

    @cached_property
    def pure_t_group(self):
        return self.get_pure_translations()

    @cached_property
    def is_primitive(self):
        return self._is_primitive()

    @cached_property
    def sog(self):
        return self.get_spin_only()

    @cached_property
    def _configuration_data(self):
        """中间属性，用于解包 conf 和 sog_direction"""
        return self.get_configuration()

    @property
    def conf(self):
        return self._configuration_data[0]

    @property
    def sog_direction(self):
        return self._configuration_data[1]

    @cached_property
    def nssg(self):
        return self.get_nssg()

    @cached_property
    def n_spin_translation_group(self):
        return self.get_nontrivial_spin_translation_group()

    # =========================================================================
    # G0 / L0 / Group 关系相关
    # =========================================================================

    @cached_property
    def G0_ops(self):
        return [[i[1], i[2]] for i in self.nssg]

    @cached_property
    def L0_ops(self):
        return [[i[1], i[2]] for i in self.nssg if np.allclose(np.eye(3), i[0], atol=0.1)]

    @cached_property
    def itik(self):
        return check_divisible(len(self.G0_ops), len(self.L0_ops))

    @cached_property
    def ik(self):
        return check_divisible(len(self.n_spin_translation_group), len(self.pure_t_group))

    @cached_property
    def it(self):
        return check_divisible(self.itik, self.ik)

    @cached_property
    def spin_part_point_ops(self):
        return deduplicate_matrix_pairs([i[0] for i in self.ops], tol=0.1)

    @cached_property
    def n_spin_part_point_ops(self):
        return deduplicate_matrix_pairs([i[0] for i in self.nssg], tol=0.1)

    # --- G0 Info ---
    @cached_property
    def _G0_info_data(self):
        """内部方法，计算所有G0相关数据并返回字典，避免副作用"""
        dataset = get_space_group_from_operations(self.G0_ops)

        G0_symbol = dataset.international
        G0_num = dataset.number

        if 74 < dataset.number < 195:
            constraint = 'a=b'
        elif dataset.number >= 195:
            constraint = 'a=b=c'
        else:
            constraint = None

        transformation_to_G0std_id = dataset.transformation_matrix
        origin_shift_to_G0std_id = dataset.origin_shift

        transformation_to_G0std = np.linalg.inv(integerize_matrix(
            np.linalg.inv(dataset.transformation_matrix), mod='col', constraint=constraint))

        origin_shift_to_G0std = transformation_to_G0std @ np.linalg.inv(
            transformation_to_G0std_id) @ origin_shift_to_G0std_id

        if in_space_group([transformation_to_G0std, origin_shift_to_G0std], self.G0_ops, tol=1e-4):
            transformation_to_G0std = np.eye(3)
            origin_shift_to_G0std = np.array([0, 0, 0])

        return {
            'symbol': G0_symbol,
            'num': G0_num,
            'trans_to_std_id': transformation_to_G0std_id,
            'origin_shift_to_std_id': origin_shift_to_G0std_id,
            'trans_to_std': transformation_to_G0std,
            'origin_shift_to_std': origin_shift_to_G0std
        }

    # 将 G0 数据暴露为属性，保持 API 一致
    @property
    def G0_symbol(self):
        return self._G0_info_data['symbol']

    @property
    def G0_num(self):
        return self._G0_info_data['num']

    @property
    def transformation_to_G0std_id(self):
        return self._G0_info_data['trans_to_std_id']

    @property
    def origin_shift_to_G0std_id(self):
        return self._G0_info_data['origin_shift_to_std_id']

    @property
    def transformation_to_G0std(self):
        return self._G0_info_data['trans_to_std']

    @property
    def origin_shift_to_G0std(self):
        return self._G0_info_data['origin_shift_to_std']

    # --- L0 Info ---
    @cached_property
    def _L0_info_data(self):
        dataset = get_space_group_from_operations(self.L0_ops)
        return {
            'symbol': dataset.international,
            'num': dataset.number,
            'trans_to_std': dataset.transformation_matrix,
            'origin_shift_to_std': dataset.origin_shift
        }

    @property
    def L0_symbol(self):
        return self._L0_info_data['symbol']

    @property
    def L0_num(self):
        return self._L0_info_data['num']

    @property
    def transformation_to_L0std(self):
        return self._L0_info_data['trans_to_std']

    @property
    def origin_shift_to_L0std(self):
        return self._L0_info_data['origin_shift_to_std']

    # --- Index ---
    @cached_property
    def index(self):
        if self._input_index:
            return self._input_index
        return self._get_ssg_index_from_ops()

    # --- Transformations ---
    @cached_property
    def G0std_L0std_transformation(self):
        return self.transformation_to_L0std @ np.linalg.inv(self.transformation_to_G0std_id)

    @cached_property
    def G0std_L0std_origin_shift(self):
        return normalize_vector_to_zero(
            -self.transformation_to_L0std @ np.linalg.inv(
                self.transformation_to_G0std_id) @ self.origin_shift_to_G0std_id + self.origin_shift_to_L0std,
            atol=1e-10
        )

    # =========================================================================
    # Point Group & Configuration Identification
    # =========================================================================

    @cached_property
    def _n_spin_part_pg_info(self):
        return identify_point_group(self.n_spin_part_point_ops)

    @property
    def n_spin_part_point_group_symbol_hm(self):
        return self._n_spin_part_pg_info[0]

    @property
    def n_spin_part_std_transformation(self):
        return self._n_spin_part_pg_info[2]

    @property
    def n_spin_part_point_group_symbol_s(self):
        return self._n_spin_part_pg_info[4]

    @cached_property
    def _spin_part_pg_info(self):
        if self.conf != 'Collinear':
            return identify_point_group(self.spin_part_point_ops)
        elif self.conf == 'Collinear':
            # 返回 dummy data 结构与 identify_point_group 一致，便于解包
            if len(self.sog) == 4:
                return ('∞m', [], np.eye(3), [], 'C∞v')
            elif len(self.sog) == 8:
                return ('∞/mm', [], np.eye(3), [], 'D∞h')
            else:
                raise ValueError('Collinear spin point group identification error')
        else:
            raise ValueError('Configuration identification error')

    @property
    def spin_part_point_group_symbol_hm(self):
        return self._spin_part_pg_info[0]

    @property
    def sppg_ops_info(self):
        return self._spin_part_pg_info[1]

    @property
    def spin_part_std_transformation(self):
        return self._spin_part_pg_info[2]

    @property
    def sppg_generators_index(self):
        return self._spin_part_pg_info[3]

    @property
    def spin_part_point_group_symbol_s(self):
        return self._spin_part_pg_info[4]

    @cached_property
    def spin_part_std_cartesian_transformation(self):
        return np.array([[1, -1 / 2, 0], [0, np.sqrt(3) / 2, 0], [0, 0, 1]]) @ np.linalg.inv(
            self.spin_part_std_transformation)

    # =========================================================================
    # K-Path & Arithmetic Class Info
    # =========================================================================

    @cached_property
    def ncnssg(self):
        return self.get_non_centered_nssg_ops()

    @cached_property
    def gspg(self):
        return self.get_general_spin_point_group_operations()

    @cached_property
    def _effective_PG_data(self):
        return self.get_effective_PG_operations()

    @property
    def eMPG(self):
        return self._effective_PG_data[0]

    @property
    def ekPG(self):
        return self._effective_PG_data[1]

    @cached_property
    def _acc_info_data(self):
        return get_arithmetic_crystal_class_from_ops([[i, j[1]] for j in self.pure_t_group for i in self.ekPG])

    @property
    def acc(self):
        return self._acc_info_data[0]

    @property
    def acc_primitive_trans(self):
        return self._acc_info_data[1]

    @property
    def acc_primitive_origin_shift(self):
        return self._acc_info_data[2]

    @property
    def kpath_info(self):
        return self._acc_info_data[3]

    @cached_property
    def acc_num(self):
        from findspingroup.data import ARITHMETIC_CRYSTAL_CLASS
        mapper = {value: key for key, value in ARITHMETIC_CRYSTAL_CLASS.SYMMORPHIC_SPACE_GROUPNUM__ACCSYMBOL.items()}
        if self.acc in mapper:
            return mapper[self.acc]
        else:
            raise ValueError('arithmetic_crystal_class error')

    @cached_property
    def cptrans(self):
        from findspingroup.data import ARITHMETIC_CRYSTAL_CLASS
        if self.acc_num in ARITHMETIC_CRYSTAL_CLASS.COMPLEXACC:
            return np.array(ARITHMETIC_CRYSTAL_CLASS.CONVENTIONAL_PRIMITIVE_TRANSFORMATIONS[
                                ARITHMETIC_CRYSTAL_CLASS.COMPLEXACC[self.acc_num]])
        else:
            return np.eye(3)

    @cached_property
    def _kpoints_data(self):
        from findspingroup.data import ARITHMETIC_CRYSTAL_CLASS
        k_conv = ARITHMETIC_CRYSTAL_CLASS.ACC_K_POINTS_CONVENTIONAL[self.acc_num]
        k_prim = ARITHMETIC_CRYSTAL_CLASS.ACC_K_POINTS_PRIMITIVE[self.acc_num]

        k_sym_c, k_val_c = zip(*k_conv)
        k_sym_p, k_val_p = zip(*k_prim)

        k_label, k_prim_str = zip(*(parse_label_and_value(i) for i in k_sym_p))
        return {
            'conv_sym': k_sym_c, 'conv_val': k_val_c,
            'prim_sym': k_sym_p, 'prim_val': k_val_p,
            'label': k_label, 'prim_str': k_prim_str
        }

    @property
    def kpoints_symbol_conventional(self):
        return self._kpoints_data['conv_sym']

    @property
    def kpoints_conventional(self):
        return self._kpoints_data['conv_val']

    @property
    def kpoints_symbol_primitive(self):
        return self._kpoints_data['prim_sym']

    @property
    def kpoints_primitive(self):
        return self._kpoints_data['prim_val']

    @property
    def kpoints_label(self):
        return self._kpoints_data['label']

    @property
    def kpoints_primitive_string(self):
        return self._kpoints_data['prim_str']

    @cached_property
    def little_groups(self):
        return self.get_little_groups()

    @cached_property
    def little_groups_symbols(self):
        return self.get_little_groups_symbols()

    @cached_property
    def is_spinsplitting(self):
        return self.is_spin_splitting()

    @cached_property
    def KPOINTS(self):
        return self.get_KPOINTS()

    @cached_property
    def is_PT(self):
        return self._is_PT()

    # =========================================================================
    # 原始逻辑方法 (静态/实例方法保持不变)
    # =========================================================================

    @staticmethod
    def is_close_matrix_pair(pair1, pair2, tol=1e-5):
        if len(pair1) != len(pair2):
            raise ValueError("Compare two vectors of different lengths.")
        for i, j in enumerate(pair1):
            if not np.allclose(pair1[i], pair2[i], atol=tol):
                return False
        return True

    @staticmethod
    def has_op(target_op, operations, tol=1e-5):
        for op in operations:
            if SpinSpaceGroup.is_close_matrix_pair(op, target_op, tol):
                return True
        return False

    def _is_PT(self):
        for op in self.ops:
            if np.allclose(-np.eye(3), op[1], atol=self.tol) and np.allclose(-np.eye(3), op[0], atol=self.tol):
                return True
        return False

    def get_KPOINTS(self):
        spin_splitting_info = [(self.kpoints_label[i], self.kpoints_primitive_string[i], True)
                               if j == 'spin splitting' else (self.kpoints_label[i], self.kpoints_primitive_string[i],
                                                              False)
                               for i, j in enumerate(self.is_spinsplitting)]
        matcher = BrillouinZoneMatcher(spin_splitting_info)
        low_symm_indices = find_uvw_whole_string(self.kpoints_symbol_primitive)
        extra_point_info = [(self.kpoints_primitive[ind], self.kpoints_label[ind]) for ind in low_symm_indices]
        return write_kpoints(self.kpath_info, matcher, extra_kpoints=extra_point_info)

    def get_little_groups_symbols(self):
        latex_symbols = []
        for index, little_group in enumerate(self.little_groups):
            spin_part = deduplicate_matrix_pairs([np.array(op[0]) for op in little_group])
            real_part = deduplicate_matrix_pairs([np.array(op[1]) for op in little_group])
            spin_info = identify_point_group(spin_part)
            real_info = identify_point_group(real_part)

            if self.conf == 'Collinear':
                t_count = 0
                for op in little_group:
                    if np.allclose(np.array(op[1]), np.eye(3), self.tol):
                        t_count += 1
                if t_count == 2:
                    spin_only_symbol = '^{\\infty }1'
                elif t_count == 4:
                    spin_only_symbol = '^{\\infty m}1'
                elif t_count == 8:
                    spin_only_symbol = '^{\\infty /mm}1'
                else:
                    raise ValueError(
                        f'Wrong spin translation group of k little group {self.kpoints_symbol_primitive[index]}')  # Fixed symbol reference
            else:
                general_spin_only = []
                for op in little_group:
                    if np.allclose(np.array(op[1]), np.eye(3), self.tol):
                        general_spin_only.append(np.array(op[0]))
                pg_symbol = identify_point_group(general_spin_only)[0]
                if pg_symbol != '1':
                    spin_only_symbol = f"^{{{pg_symbol}}}1"
                else:
                    spin_only_symbol = ''

            # match spin op
            spin_generators = []
            for index_g in real_info[3]:  # fixed loop variable name clash
                for op in little_group:
                    if np.allclose(np.array(op[1]), real_info[1][index_g][0], self.tol):
                        spin_generators.append(op[0])
                        break

            spin_generators_symbols = []
            for spin_op in spin_generators:
                for op in spin_info[1]:
                    if np.allclose(np.array(op[0]), spin_op, self.tol):
                        spin_generators_symbols.append(op[2])

            latex = ''
            if bool(re.search(r'/', real_info[0])):
                count = 0
                for i, index_g in enumerate(real_info[3]):
                    if count == 1:
                        latex = latex + '/' + '^{' + spin_generators_symbols[i] + '}' + real_info[1][index_g][2]
                    else:
                        latex = latex + '^{' + spin_generators_symbols[i] + '}' + real_info[1][index_g][2]
                    count += 1
                latex = latex + spin_only_symbol
            else:
                for i, index_g in enumerate(real_info[3]):
                    latex = latex + '^{' + spin_generators_symbols[i] + '}' + real_info[1][index_g][2]
                latex = latex + spin_only_symbol

            latex_symbols.append(latex)
        return latex_symbols

    def get_little_groups(self):
        k_little_groups = []
        if self.is_primitive:
            kpoints = self.kpoints_primitive
        else:
            kpoints = self.kpoints_conventional

        if self.cptrans is None or np.allclose(self.cptrans, np.eye(3)):
            # Simplified logic check: if P center lattice not complex lattice
            for k_point in kpoints:
                little_group = []
                for op in self.gspg:
                    eop = np.linalg.det(op[0]) * np.array(op[1])
                    target_kpoint = eop @ np.array(k_point) % 1
                    diff = getNormInf(np.array(k_point) % 1, target_kpoint)
                    if diff < self.tol:
                        little_group.append(op)
                k_little_groups.append(little_group)
        else:
            # if complex center lattice
            for k_point in kpoints:
                little_group = []
                for op in self.gspg:
                    eop = np.linalg.det(op[0]) * np.array(op[1])
                    if self.is_primitive:
                        target_kpoint = eop @ np.array(k_point) % 1
                        diff = getNormInf(np.array(k_point) % 1, target_kpoint)
                        if diff < self.tol:
                            little_group.append(op)
                    else:
                        # Check complex lattice condition
                        if getNormInf(self.cptrans.T @ np.array(k_point) % 1, (
                                np.linalg.inv(self.cptrans) @ eop @ self.cptrans @ self.cptrans.T @ np.array(
                                k_point) % 1)) < self.tol:
                            little_group.append(op)
                k_little_groups.append(little_group)
        return k_little_groups

    def _is_primitive(self):
        if len(self.pure_t_group) > 1:
            return False
        else:
            return True

    def get_effective_PG_operations(self):
        effective_magnetic_point_group = []
        effective_k_point_group = []
        for i in self.gspg:
            if abs(np.linalg.det(i[0]) - 1) < self.tol:
                effective_magnetic_point_group.append([np.eye(3), i[1]])
                effective_k_point_group.append(np.array(i[1]))
            elif abs(np.linalg.det(i[0]) + 1) < self.tol:
                effective_magnetic_point_group.append([-np.eye(3), i[0]])
                effective_k_point_group.append(-1 * np.array(i[1]))
            else:
                raise ValueError('tolerance error when getting general spin point group')
        return deduplicate_matrix_pairs(effective_magnetic_point_group, tol=self.tol), deduplicate_matrix_pairs(
            effective_k_point_group, tol=self.tol)

    def get_general_spin_point_group_operations(self)->'GeneralizedSpinPointGroup':
        return GeneralizedSpinPointGroup(deduplicate_matrix_pairs([[i[0], i[1]] for i in self.ops], tol=self.tol))

    def get_non_centered_nssg_ops(self):
        eq_class = []
        num_ops = len(self.nssg)
        assigned = [False] * num_ops
        for i in range(num_ops):
            if assigned[i]: continue
            class_i = []
            for j in self.n_spin_translation_group:
                check_op = self.nssg[i] @ j
                for k in range(num_ops):
                    if assigned[k]: continue
                    if self.nssg[k].is_same_with(check_op, self.tol):
                        class_i.append(k)
                        assigned[k] = True
                        break
            eq_class.append(class_i)
        if len(set([len(i) for i in eq_class])) != 1:
            raise ValueError('Wrong number of co-set. Check tolerance!')
        non_centered_nssg_ops = []
        for i in eq_class:
            non_centered_nssg_ops.append(self.nssg[i[0]])
        return non_centered_nssg_ops

    def get_nontrivial_spin_translation_group(self):
        nontrivial_spin_translation_group = []
        for i in self.nssg:
            if np.allclose(i[1], np.eye(3), self.tol):
                nontrivial_spin_translation_group.append(i)
        return nontrivial_spin_translation_group

    def get_nssg(self):
        nssg = []
        if self.conf == 'Collinear':
            for i in self.ops:
                if np.allclose(i[0], -np.eye(3), self.tol) or np.allclose(i[0], np.eye(3), self.tol):
                    nssg.append(i)
        elif self.conf == 'Coplanar':
            for i in self.ops:
                if np.linalg.det(i[0]) > 0:
                    nssg.append(i)
        else:
            nssg = self.ops
        return nssg

    def get_configuration(self):
        if len(self.sog) == 8 or len(self.sog) == 4:
            for operations in self.sog:
                if not np.allclose(operations[0], np.eye(3), atol=0.1) and abs(np.linalg.det(operations[0]) - 1) < 1e-2:
                    eigvals, eigvecs = np.linalg.eig(operations[0])
                    direction = eigvecs[:, np.isclose(eigvals, 1.0, atol=0.1)].real
                    return 'Collinear', direction
        if len(self.sog) == 2:
            for operations in self.sog:
                if not np.allclose(operations[0], np.eye(3), atol=0.1):
                    eigvals, eigvecs = np.linalg.eig(operations[0])
                    direction = eigvecs[:, np.isclose(eigvals, -1.0, atol=0.1)].real
                    return 'Coplanar', direction
        if len(self.sog) == 1:
            return 'Noncoplanar', None
        raise ValueError('Wrong spin only groups. Check tolerance!')

    def get_spin_translation_group(self):
        spin_translation_group = []
        for op in self.ops:
            if np.allclose(op[1], np.eye(3), self.tol):
                spin_translation_group.append(op)
        return deduplicate_matrix_pairs(spin_translation_group)

    def get_pure_translations(self):
        pure_translations = []
        for op in self.spin_translation_group:
            if np.allclose(op[0], np.eye(3), self.tol):
                pure_translations.append([op[1], op[2]])
        return pure_translations

    def get_spin_only(self):
        spin_only_group = []
        for i in self.spin_translation_group:
            if np.allclose(i[2], np.zeros(3), atol=1e-5):
                spin_only_group.append(i)
        return spin_only_group

    def build_multiplication_table(self):
        n = len(self.ops)
        table = np.empty((n, n), dtype=int)
        for i in range(n):
            for j in range(n):
                product = self.ops[i] @ self.ops[j]
                for k in range(n):
                    if product == self.ops[k]:
                        table[i, j] = k
                        break
                else:
                    raise ValueError("Product not found in group operations.")
        return table

    def _get_ssg_index_from_ops(self) -> str:
        G0 = self.G0_num
        L0 = self.L0_num
        it = self.it
        ik = self.ik
        return f"{G0}.{L0}.{it}.{ik}"

    def get_index(self):
        return self.index

    def __hash__(self):
        return hash(self.index)

    def __len__(self):
        return len(self.ops)

    def __repr__(self):
        lines = [f"<SpinSpaceGroup #{self.index} '>"]
        for i, group in enumerate(self.ops):
            lines.append(f"Group {i}:")
            mats = [np.atleast_2d(np.array(m)).reshape(3, -1) for m in group]
            mat_strs = [np.array2string(m,
                                        formatter={'float_kind': lambda x: f"{x:5.3f}"},
                                        separator=' ', suppress_small=True).splitlines()
                        for m in mats]
            widths = [max(len(line) for line in s) for s in mat_strs]
            n_rows = max(len(s) for s in mat_strs)
            for r in range(n_rows):
                row_parts = []
                for j, s in enumerate(mat_strs):
                    content = s[r] if r < len(s) else " " * widths[j]
                    row_parts.append(content.ljust(widths[j]))
                    if j == len(mat_strs) - 2:
                        row_parts.append("   |   ")
                    else:
                        row_parts.append("   ")
                lines.append("".join(row_parts))
            lines.append("")  # 空行分组
        return "\n".join(lines)

    def get_generators(self):
        return find_group_generators(self.ops)

    def to_spin_point_group(self):
        spg_ops = [op.to_spg_op() for op in self.ops]
        return GeneralizedSpinPointGroup(spg_ops)

    def transform(self, transformation_matrix, origin_shift, frac=True, all_trans=True):
        transformation_matrix_inv = np.linalg.inv(transformation_matrix)
        if frac:
            translations = integer_points_in_new_cell(transformation_matrix_inv.T)
        else:
            translations = [np.zeros(3)]
        if not all_trans:
            translations = [np.zeros(3)]

        new_ops = []
        for op in [[i[0], i[1], i[2] + np.array(j)] for i in self.ops for j in translations]:
            new_rotation = transformation_matrix @ op[1] @ transformation_matrix_inv
            if frac:
                new_translation = normalize_vector_to_zero(
                    ((np.eye(3) - new_rotation) @ origin_shift + transformation_matrix @ op[2]), atol=1e-4)
            else:
                new_translation = ((np.eye(3) - new_rotation) @ origin_shift + transformation_matrix @ op[2])
            new_op = SpinSpaceGroupOperation(op[0], new_rotation, new_translation)
            new_ops.append(new_op)

        # 返回新实例，它会自动惰性计算自己的属性
        return SpinSpaceGroup(new_ops)

    def transform_spin(self, spin_transformation_matrix):
        spin_transformation_matrix_inv = np.linalg.inv(spin_transformation_matrix)
        new_ops = []
        for op in self.ops:
            new_spin_rotation = spin_transformation_matrix @ op[0] @ spin_transformation_matrix_inv
            new_op = SpinSpaceGroupOperation(new_spin_rotation, op[1], op[2])
            new_ops.append(new_op)
        return SpinSpaceGroup(new_ops)

    def get_attributes_from_database(self):
        attributes = {
            "crystal_system": "cubic",
            "point_group": "m-3m",
            "lattice_type": "P",
        }
        return attributes

    def is_spin_splitting(self):
        spinsplitting = []
        for little_group in self.little_groups:
            spinmatrices = np.vstack(deduplicate_matrix_pairs([op[0] - np.eye(3) for op in little_group], tol=self.tol))
            if all(abs(x) > 1e-3 for x in np.linalg.svd(spinmatrices.astype(np.float32))[1]):
                spinsplitting.append('no spin splitting')
            else:
                spinsplitting.append('spin splitting')
        return spinsplitting


class GeneralizedSpinPointGroup:
    def __init__(self, ops):
        self.ops = ops

    def __getitem__(self, index):
        return self.ops[index]

    def __repr__(self):
        lines = [f"<GeneralizedSpinPointGroup with {len(self.ops)} operations>"]
        for i, group in enumerate(self.ops):
            lines.append(f"Group {i}:")
            mats = [np.atleast_2d(np.array(m)).reshape(3, -1) for m in group]
            mat_strs = [np.array2string(m,
                                        formatter={'float_kind': lambda x: f"{x:5.3f}"},
                                        separator=' ', suppress_small=True).splitlines()
                        for m in mats]
            widths = [max(len(line) for line in s) for s in mat_strs]
            n_rows = max(len(s) for s in mat_strs)
            for r in range(n_rows):
                row_parts = []
                for j, s in enumerate(mat_strs):
                    content = s[r] if r < len(s) else " " * widths[j]
                    row_parts.append(content.ljust(widths[j]))
                    if j == len(mat_strs) - 2:
                        row_parts.append("   |   ")
                    else:
                        row_parts.append("   ")
                lines.append("".join(row_parts))
            lines.append("")
        return "\n".join(lines)

    @property
    def effective_magnetic_point_group(self):
        return self.get_effective_magnetic_point_group()

    @property
    def empg_symbol(self):
        return self.get_empg_symbol()

    def get_effective_magnetic_point_group(self):
        effective_magnetic_point_group = []
        for i in self.ops:
            if abs(np.linalg.det(i[0]) - 1) < 1e-2:
                effective_magnetic_point_group.append([1, i[1]])
            elif abs(np.linalg.det(i[0]) + 1) < 1e-2:
                effective_magnetic_point_group.append([-1, i[1]])
            else:
                raise ValueError('tolerance error when getting general spin point group')

        return deduplicate_matrix_pairs(effective_magnetic_point_group, tol=1e-3)


    def get_empg_symbol(self):
        empg_ops = self.effective_magnetic_point_group
        for i in empg_ops:
            if i[0]==-1 and np.allclose(i[1],np.eye(3),atol=1e-5):
                try:
                    effective_space_rotation = deduplicate_matrix_pairs([_[1] for _ in empg_ops], tol=1e-5)
                    empg_symbol=get_space_group_from_operations([[j,np.array([0,0,0])] for j in effective_space_rotation]).pointgroup+"1'"
                except:
                    empg_symbol=None
                return empg_symbol
        empg_info = get_magnetic_space_group_from_operations([[i[0],i[1],np.array([0,0,0])] for i in empg_ops])
        return empg_info['mpg_symbol']




