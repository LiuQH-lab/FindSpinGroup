import copy
import math
import re
from fractions import Fraction
from seekpath import get_path

import numpy as np

from spglib import SpglibDataset, get_symmetry_dataset, SpglibMagneticDataset, get_magnetic_symmetry_dataset
from findspingroup.data.MSGMPG_DB import MSG_INT_TO_BNS, BNS_TO_OG_NUM, OG_NUM_TO_MPG
from findspingroup.utils import SG_HALL_MAPPING
from findspingroup.utils.matrix_utils import normalize_vector_to_zero

np.set_printoptions(suppress=True)


def get_element_order(element):
    # element := 3*3 np.array

    errorblocker = 0
    order = 1
    temp = element

    while not np.allclose(temp, np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), atol=0.1):
        # tolerance can be large
        temp = temp @ element
        order = order + 1
        errorblocker = errorblocker + 1
        if errorblocker > 61:
            raise ValueError('error cannot find the order')
    return order

def rotation_angle(R,axis,eigenvals):
    v1 = np.random.rand(3)
    axis = axis
    v1 -= v1.dot(axis) * axis  # get perpendicular vector v1
    v1 /= np.linalg.norm(v1)
    v2 = R @ v1
    cross = np.cross(v1, v2)
    sign = np.sign(np.dot(axis, cross))

    a = [val for val in eigenvals if val.imag > 0.01]
    angle = np.arccos(a[0].real)
    if sign < 0:
        angle = 2*3.14159265357-angle  # countwise rotation

    return angle

def times_of_rotation(rotation_angle):
    """
        2*pi*m/n = rotation_angle
        m/n = rotation_angle/2pi
    """

    m_n = rotation_angle/(2*3.14159265357)
    f = Fraction(m_n).limit_denominator(30)
    m, n = f.numerator, f.denominator
    return m, n


def costheta(v1, v2):
    """计算两个向量夹角的余弦值 (假设 v1, v2 未归一化)"""
    norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norm_product == 0:
        return 0
    return np.dot(v1, v2) / norm_product


def find_rotation(operations, rotation_times, axis, perp_axis=None, exclude=None, improper=False):

    """
    寻找符合几何约束的旋转操作。

    Parameters:
    -----------
    operations : list
        操作列表，格式: [..., ..., ..., vector, (power, order), det]
    rotation_times : int
        目标旋转阶数 (n)
    axis : array_like
        目标轴方向 (target axis)
    perp_axis : array_like, optional
        参考垂直轴。如果提供，将用于确定右手系方向。
    exclude : array_like, optional
        需要排除的轴（通常是已确定的主轴）。
    improper : bool
        是否寻找非正规旋转 (improper rotation)。

    Returns:
    --------
    (index, direction) : tuple
        找到的操作索引和校正后的方向向量。
    """

    # --- 物理阈值常量 ---
    TOL_PARALLEL = 1e-2  # 判定平行的容差
    TOL_CHIRALITY = 1e-4  # 判定混合积是否为0的容差
    MIN_DEVIATION = 1.000001  # 初始最小偏差值

    best_op_index = None
    best_direction = None
    min_deviation = MIN_DEVIATION

    # 目标行列式值: Proper (+1), Improper (-1)
    target_det = -1 if improper else 1

    for i, op in enumerate(operations):
        # 1. 解包数据，提高可读性
        op_vec = np.array(op[3])
        op_rot_info = op[4]  # (power, order)
        op_det = op[5]

        # 2. 基础属性筛选
        if op_det != target_det:
            continue
        if op_rot_info is None or op_rot_info[1] != rotation_times:
            continue

        # 3. 几何筛选：排除轴 (Exclude)
        # 如果当前轴平行于 exclude 轴，直接跳过
        if exclude is not None:
            if abs(abs(costheta(exclude, op_vec)) - 1) < TOL_PARALLEL:
                continue

        # 4. 几何筛选：垂直轴 (Perp_axis)
        # 原逻辑：如果太过于平行于 perp_axis (即不垂直)，跳过
        # 这里保留你原代码的意图: abs(cos) > 0.8 意味着夹角 < 36度，显然不够垂直
        if perp_axis is not None:
            if abs(costheta(perp_axis, op_vec)) > 0.8:
                continue

        # 5. 确定候选方向
        # 计算轴与目标的对齐程度
        cos_val = costheta(op_vec, axis)

        # 我们只关心当前这个轴是否比之前找到的更接近 target axis
        current_deviation = abs(abs(cos_val) - 1)
        if current_deviation > min_deviation:
            continue

        # 生成候选列表：可能是原向量，也可能是反向量
        # 根据你的逻辑，只考虑 power=1 (基转动) 或 power=n-1 (逆转动)
        candidates = []

        # 检查正向 (+op_vec)
        if op_rot_info[0] == 1:
            candidates.append(op_vec)

        # 检查反向 (-op_vec)，通常对应 op_rot_info[0] == order - 1
        # 如果你的数据中包含逆操作，这里可以放宽条件，或者保持你原有的逻辑
        if op_rot_info[1] - op_rot_info[0] == 1:
            candidates.append(-op_vec)

        for candidate_vec in candidates:
            # --- 关键修改：统一的右手系检查 ---
            is_valid = True

            # 如果提供了 exclude 和 perp_axis，我们必须检查手性
            if exclude is not None and perp_axis is not None:
                # 计算混合积 (exclude x candidate) · perp
                # 几何意义：如果 > 0，则 (exclude, candidate, perp) 构成右手系
                # 注意：这里向量的顺序决定了谁是x, y, z。
                # 假设顺序是: Exclude(主轴) -> Candidate(副轴) -> Perp(第三轴)
                mixed_product = np.dot(perp_axis, np.cross(exclude, candidate_vec))

                if mixed_product < TOL_CHIRALITY:
                    # 如果混合积为负(左手系) 或 为0(共面)，则不仅是不符合右手系，也可能是方向错了
                    is_valid = False

            # 如果只需要垂直检查，不需要手性（比如只给了 perp 但没给 exclude）
            elif perp_axis is not None:
                # 比如 Cubic 系统中避免选到不想要的对角线方向
                # 原代码逻辑: costheta(perp_axis, vec) < -0.1 continue
                if costheta(perp_axis, candidate_vec) < -0.1:
                    is_valid = False

            # 只有 exclude 没有 perp 的情况 (原代码逻辑)
            elif exclude is not None:
                if costheta(exclude, candidate_vec) < 0.1:  # 稍微偏向正向
                    is_valid = False

            # --- 更新最佳结果 ---
            if is_valid:
                # 重新计算该 candidate 的具体偏差
                final_cos = costheta(candidate_vec, axis)
                # 只有当它确实指向我们想要的方向 (cos > 0) 且偏差更小时才更新
                # 注意：如果上面通过了右手系检查，但方向和 target axis 反了，
                # 说明 target axis 本身可能不是右手系的建议方向，这里需要权衡。
                # 但通常我们寻找的是 "最接近 target axis 且满足右手系" 的操作。

                this_deviation = abs(final_cos - 1)  # 我们希望 cos 接近 +1

                # 如果这是目前找到的最优解
                if this_deviation < min_deviation:
                    min_deviation = this_deviation
                    best_op_index = i
                    best_direction = candidate_vec

    if best_op_index is None:
        raise ValueError(f"No rotation found for order={rotation_times}, improper={improper}")

    return best_op_index, best_direction


def find_mirror(operations,axis,perp_axis=None,exclude=None,cubic=None):
    index = None
    distance = 1.001
    for i,op in enumerate(operations):
        if op[2] == 'm':
            if perp_axis is not None: # skip those not perp

                if abs(costheta(perp_axis,op[3])) > 0.9:
                    continue
            if exclude is not None:
                if abs(abs(costheta(exclude,op[3]))-1) < 1e-2:
                    continue
            if cubic is not None:
                if cubic is True and exclude is not None and abs(costheta(exclude,op[3])) < 0.65:
                    continue

                if cubic is not True and (abs(costheta(cubic,op[3])) > 0.65  or abs(costheta(cubic,op[3])) < 0.4): # exclude the nearest mirror of high-symmetry axis
                    continue
            temp_distance = costheta(op[3],axis)+0.001
            if abs(abs(temp_distance)-1)<=distance:
                if np.sign(temp_distance) < 0:
                    if exclude is not None and perp_axis is not None:
                        t = costheta(perp_axis,exclude)
                        tempc= costheta(perp_axis, np.cross(exclude, -op[3]))
                        if tempc < 0:
                            continue
                    if perp_axis is not None and costheta(perp_axis, -op[3]) < -0.1:  # for cubic system
                        continue
                    distance = abs(abs(temp_distance)-1)
                    index = i
                    direction = -op[3]
                else:
                    if exclude is not None and perp_axis is not None and abs(costheta(exclude, op[3])) < 1e-2 and costheta(perp_axis,np.cross(exclude,op[3])) > 0:
                        distance = abs(abs(temp_distance) - 1)
                        index = i
                        direction = op[3]
                        continue
                    if exclude is not None and perp_axis is not None and abs(costheta(exclude, op[3])) < 1e-2 and costheta(perp_axis,np.cross(exclude,-op[3])) > 0:
                        distance = abs(abs(temp_distance) - 1)
                        index = i
                        direction = -op[3]
                        continue
                    if exclude is not None and perp_axis is not None and costheta(perp_axis, np.cross(exclude, op[3])) < 0:
                        continue
                    if perp_axis is not None and costheta(perp_axis, op[3]) < -0.1:  # for cubic system
                        continue
                    distance = abs(abs(temp_distance)-1)
                    index = i
                    direction = op[3]
            if cubic is not None and cubic is not True and index == i:
                if costheta(cubic,op[3]) < 0:
                    direction = -op[3]
                else:
                    direction = op[3]
    if index is None:
        raise ValueError("No mirror found")
    return index, direction


def reverse_direction(operations, direction):
    """
        reverse direction
        operations : op_order_type_direction_addition_det
    """
    for op in operations:
        if op[3] is not None:
            if abs(np.dot(op[3],direction) + 1) < 1e-2: #  opposite direction
                op[3] = -op[3] # reverse direction
                if op[4] != None: # only for rotations
                    op[4] = [op[4][1]-op[4][0],op[4][1]] # [n-m,n]
    return operations



def identify_point_group(point_group_matrices,tol=1e-2):
    """

        input : point_group_matrices [op,...]

        return symbol_HM, op_symbols(type + direction), transformation matrix, generators_index, symbol_S
    """

    # Step 1 : determination of ops

    op_order_type_direction_addition_det = []
    for op in point_group_matrices:
        order = get_element_order(op)

        if order == 1: # map 1
            op_order_type_direction_addition_det.append([op,order,'1',None,None,1])

        eigvals, eigvecs = np.linalg.eig(op.astype(np.float64))
        eigvecs = eigvecs.T

        if order > 2 : # map n or -n
            for i,val in enumerate(eigvals):
                if abs(val.imag) < tol: # find the direction
                    if abs(val.real - 1) < tol: # n
                        axis = eigvecs[i].real
                        angle = rotation_angle(op,axis,eigvals)
                        m , n = times_of_rotation(angle)
                        op_order_type_direction_addition_det.append([op,order,str(n),axis,[m,n],1])  # rotate around axis m times of 2*pi/n
                        break

                    if abs(val.real + 1) < tol: # -n
                        axis = eigvecs[i].real
                        angle = rotation_angle(-op,axis,-eigvals)   #   multiply -1 to op
                        m , n = times_of_rotation(angle)
                        op_order_type_direction_addition_det.append([op,order,str(-n),axis,[m,n],-1])  # rotate around axis m times of 2*pi/n
                        break
                if i == 2:
                    raise ValueError('can not find eigenvector for rotation, try another tolerance!')

        if order == 2: # -1, m, 2
            if abs(sum(eigvals) + 3) < tol: # -1
                op_order_type_direction_addition_det.append([op,order,'-1',None,None,-1])

            if abs(sum(eigvals) + 1) < tol: # 2
                for i,val in enumerate(eigvals):
                    if abs(val-1) < tol: # find direction
                        axis = eigvecs[i]
                        break
                    if i == 2:
                        raise ValueError('can not find eigenvector for rotation 2, try another tolerance!')
                op_order_type_direction_addition_det.append([op,order,'2',axis,[1,2],1])

            if abs(sum(eigvals) - 1) < tol: # m
                for i,val in enumerate(eigvals):
                    if abs(val + 1) < tol: # find norm vector
                        axis = eigvecs[i]
                        break
                    if i == 2:
                        raise ValueError('can not find norm vector for mirror, try another tolerance!')
                op_order_type_direction_addition_det.append([op,order,'m',axis,None,-1])


    # Step 2 : determination of point group
    # Step 3 : transformation matrix

    order_group = len(op_order_type_direction_addition_det)


    # change basis for metric
    metric_matrix_G = np.array(sum([op[0].T@op[0] for op in op_order_type_direction_addition_det])/order_group,dtype=np.float64)
    # metric_matrix_G = recover_metric_positive_definite_relaxed([op[0] for op in op_order_type_direction_addition_det])
    # use g = 1/|G| sum(R^T@ I @R) to recover metric
    if np.allclose(metric_matrix_G, np.eye(3), rtol=1e-2):
        P1 = np.eye(3)
        P1_inv = np.eye(3)
        operations = copy.deepcopy(op_order_type_direction_addition_det)
    else:
        P1_inv = np.linalg.cholesky(metric_matrix_G).T
        P1 = np.linalg.inv(P1_inv)
        operations = copy.deepcopy(op_order_type_direction_addition_det)
        for op in operations:
            if op[3] is not None:
                op[0] = P1_inv @ op[0] @ P1
                op[3] = P1_inv @ op[3]



    if order_group == 1:
        group_symbol = '1'
        group_symbol_S = 'C1'
        return '1',op_order_type_direction_addition_det,np.eye(3),[0],group_symbol_S


    counter_high_order_axis = []
    mirror = 0
    minues = False
    max_order = 1
    rotation2_axis = []
    improper = False

    for i in operations:
        if improper:
            pass
        else:
            if i[5] == -1:
                improper = True
        if i[1] > 2: # order > 2
            if counter_high_order_axis == []: # initialization
                counter_high_order_axis.append(i[3])
            else:
                if not any([abs(abs(costheta(i[3], _ ))-1) < tol for _ in counter_high_order_axis]):  # different axis for high order
                    counter_high_order_axis.append(i[3])
        if i[2] == '2':
            rotation2_axis.append(i[3])
        if i[2] == 'm':
            mirror = mirror + 1
        if i[2] == '-1':
            minues = True
        max_order = max(max_order, i[1])



    if len(counter_high_order_axis) > 1: #23 m-3 432 -43m m-3m I Ih
        if order_group == 12: # T
            group_symbol = '23'
            group_symbol_S = 'T'
            r1_index, direction = find_rotation(operations,
                                                      2, np.array([1, 0, 0]))
            operations = reverse_direction(operations,
                                                                     direction)
            r2_index, r2_direction = find_rotation(operations,
                                                      3, np.array([1, 1, 1]),direction)
            operations = reverse_direction(operations,
                                                                     r2_direction)
            generators = [operations[r1_index][0],operations[r2_index][0]]  # matrix
            generators_index = [r1_index,r2_index]
        if order_group == 24: #-43m 432 m-3
            if minues: # Th
                group_symbol = 'm-3'
                group_symbol_S = 'Th'

                ir_index, ir_direction = find_rotation(operations,
                                                       3, np.array([1, 1, 1]), improper=True)
                operations = reverse_direction(operations, ir_direction)

                mirror_index, m_direction = find_mirror(operations, np.array([1, 0, 0]), cubic=ir_direction)
                operations = reverse_direction(operations, m_direction)
                generators = [operations[mirror_index][0],operations[ir_index][0]]  # matrix
                generators_index = [mirror_index,ir_index]
            else:
                if mirror > 0: # Td
                    group_symbol = '-43m'
                    group_symbol_S = 'Td'
                    ir_index, ir_direction = find_rotation(operations,
                                                           4, np.array([1, 0, 0]), improper=True)
                    operations = reverse_direction(operations,
                                                                             ir_direction)
                    r_index, r_direction = find_rotation(operations,
                                                           3, np.array([1, 1, 1]),ir_direction)
                    operations = reverse_direction(operations,
                                                                             r_direction)
                    generators = [operations[ir_index][0],
                                  operations[r_index][0],
                                  operations[ir_index][0] @ operations[r_index][0] @ operations[ir_index][0]@ operations[ir_index][0] @ operations[r_index][0] @operations[ir_index][0]@operations[ir_index][0]
                                  ]  # matrix -4_100 @ 3+_111 @ 2_100 @ 3+_111 @ 2_100
                    generators_index = [ir_index,r_index,np.where([np.allclose(i[0],operations[ir_index][0] @ operations[r_index][0] @ operations[ir_index][0] @ operations[ir_index][0]@ operations[r_index][0] @operations[ir_index][0] @operations[ir_index][0],atol=1e-2) for i in operations])[0][0]]


                else:  # O
                    group_symbol = '432'
                    group_symbol_S = 'O'
                    r_index, r_direction = find_rotation(operations,
                                                           4, np.array([1, 0, 0]))
                    operations = reverse_direction(operations,
                                                                             r_direction)
                    r2_index, r2_direction = find_rotation(operations,
                                                           3, np.array([1, 1, 1]),r_direction)
                    operations = reverse_direction(operations,
                                                                             r2_direction)
                    generators = [operations[r_index][0],
                                  operations[r2_index][0],
                                  operations[r_index][0] @ operations[r2_index][0] @ operations[r_index][0]@ operations[r_index][0] @ operations[r2_index][0] @operations[r_index][0]@operations[r_index][0]
                                  ]  # matrix 4_100 @ 3+_111 @ 2_100 @ 3+_111 @ 2_100
                    generators_index = [r_index,r2_index,np.where([np.allclose(i[0],operations[r_index][0] @ operations[r2_index][0] @ operations[r_index][0]@ operations[r_index][0] @ operations[r2_index][0] @operations[r_index][0]@operations[r_index][0],rtol=1e-2) for i in operations])[0][0]]

        if order_group == 48: # Oh
            group_symbol = 'm-3m'
            group_symbol_S = 'Oh'

            ir_index, ir_direction = find_rotation(operations,
                                                   3, np.array([1, 1, 1]), improper=True)
            operations = reverse_direction(operations,ir_direction)

            mirror_index, m_direction = find_mirror(operations, np.array([1, 0, 0]),cubic=ir_direction)
            operations = reverse_direction(operations,m_direction)

            m2_index, m2_direction = find_mirror(operations, ir_direction,ir_direction,m_direction,cubic = True)
            operations = reverse_direction(operations,
                                                                     m2_direction)

            generators = [operations[mirror_index][0],
                          operations[ir_index][0],
                          operations[m2_index][0]
                          ]  # matrix
            generators_index = [mirror_index,ir_index,m2_index]

        if order_group == 60: # I
            group_symbol = '532'
            group_symbol_S = 'I'

        if order_group == 120:  # Ih
            group_symbol = '-5-32'
            group_symbol_S = 'Ih'

    if len(counter_high_order_axis) == 1: # n  or  -n             Cn Cnv Cnh Dn Dnh Dnd Sn           (n>2)
        if improper: # Cnv Cnh Dnh Dnd Sn ---without D2h C2v D2 C2h Cs C2 Ci
            if order_group == max_order: # S2n
                if (order_group / 2) % 2 == 0: # -2n    S2n
                    group_symbol = f'-{max_order}'
                    group_symbol_S = f'S{max_order}'
                    improper_rotation_index, direction = find_rotation(operations,max_order,np.array([0,0,1]),improper=True)
                    operations = reverse_direction(operations,direction)
                    generators = [operations[improper_rotation_index][0]] # matrix
                    generators_index = [improper_rotation_index]
                else:
                    if minues:
                        group_symbol = f'-{int(max_order/2)}'  # -n   S2n
                        group_symbol_S = f'S{max_order}'
                        improper_rotation_index, direction = find_rotation(operations,
                                                                           int(max_order/2), np.array([0, 0, 1]), improper=True)
                        operations = reverse_direction(operations,
                                                                                 direction)
                        generators = [operations[improper_rotation_index][0]] # matrix
                        generators_index = [improper_rotation_index]
                    else:
                        group_symbol = f'-{max_order}' # -2n  Cnh
                        group_symbol_S = f'C{int(max_order/2)}h'
                        improper_rotation_index, direction = find_rotation(operations,
                                                                           max_order, np.array([0, 0, 1]), improper=True)
                        operations = reverse_direction(operations,
                                                                                 direction)
                        generators = [operations[improper_rotation_index][0]] # matrix
                        generators_index = [improper_rotation_index]
            else:
                if len(rotation2_axis) < 2: # Cnv Ceh
                    if minues: # n/m    Cnh
                        group_symbol = f'{int(order_group/2)}/m'
                        group_symbol_S = f'C{int(order_group/2)}h'
                        rotation_index, direction = find_rotation(operations,
                                                                           int(order_group/2), np.array([0, 0, 1]))
                        operations = reverse_direction(operations,
                                                                                 direction)
                        mirror_index, m_direction = find_mirror(operations, direction)

                        generators = [operations[rotation_index][0],operations[mirror_index][0]] # matrix
                        generators_index = [rotation_index,mirror_index]
                    else:
                        if (order_group / 2) % 2 == 0: # nmm    Cnv
                            group_symbol = f'{int(order_group/2)}mm'
                            group_symbol_S = f'C{int(order_group/2)}v'
                            rotation_index, direction = find_rotation(operations,
                                                                      int(order_group / 2), np.array([0, 0, 1]))
                            operations = reverse_direction(
                                operations,
                                direction)
                            mirror_index, m_direction = find_mirror(operations, np.array([1, 0, 0]))
                            operations = reverse_direction(operations,m_direction)
                            mirror2_index, m2_direction = find_mirror(operations, m_direction,direction,exclude=m_direction)
                            operations = reverse_direction(operations,
                                                                                     m2_direction)
                            generators = [operations[rotation_index][0],operations[mirror_index][0],operations[mirror2_index][0]]
                            generators_index = [rotation_index,mirror_index,mirror2_index]



                        else: # nm  Cnv
                            if int(order_group/2)>9:
                                group_symbol = f'({int(order_group/2)})m'
                                group_symbol_S = f'C{int(order_group/2)}v'
                            else:
                                group_symbol = f'{int(order_group/2)}m'
                                group_symbol_S = f'C{int(order_group/2)}v'
                            rotation_index, direction = find_rotation(operations,
                                                                      int(order_group / 2), np.array([0, 0, 1]))
                            operations = reverse_direction(
                                operations,
                                direction)
                            mirror_index, m_direction = find_mirror(operations, np.array([1, 0, 0]))
                            operations = reverse_direction(operations,m_direction)
                            generators = [operations[rotation_index][0],operations[mirror_index][0]]
                            generators_index = [rotation_index,mirror_index]




                else: # Dnd Dnh     --- without D2h
                    if int(order_group / 4) % 2 == 0:
                        if minues: # n/mmm    Dnh
                            group_symbol = f'{int(order_group/4)}/mmm'
                            group_symbol_S = f'D{int(order_group/4)}h'
                            rotation_index, direction = find_rotation(operations,
                                                                      int(order_group / 4), np.array([0, 0, 1]))
                            operations = reverse_direction(
                                operations,
                                direction)
                            mz_index, mz_direction = find_mirror(operations, direction)
                            m_index, m_direction = find_mirror(operations, np.array([1, 0, 0]),direction)
                            operations = reverse_direction(operations,m_direction)
                            m2_index, m2_direction = find_mirror(operations, m_direction,direction,m_direction)
                            operations = reverse_direction(operations,m2_direction)
                            generators = [operations[rotation_index][0],operations[mz_index][0],operations[m_index][0],operations[m2_index][0]]
                            generators_index = [rotation_index,mz_index,m_index,m2_index]

                        else: #-(2n)2m    Dnd
                            if int(order_group/2) > 9:
                                group_symbol = f'-({int(order_group / 2)})2m'
                                group_symbol_S = f'D{int(order_group / 4)}d'
                            else:
                                group_symbol = f'-{int(order_group/2)}2m'
                                group_symbol_S = f'D{int(order_group/4)}d'
                            srotation_index, direction = find_rotation(operations,
                                                                      int(order_group / 2), np.array([0, 0, 1]),improper=True)

                            operations = reverse_direction(
                                operations,
                                direction)
                            r_index, r_direction = find_rotation(operations,
                                                                      2, np.array([1, 0, 0]), direction)
                            operations = reverse_direction(operations,r_direction)
                            m_index, m_direction = find_mirror(operations, r_direction,direction,r_direction)
                            operations = reverse_direction(operations,m_direction)
                            generators = [operations[srotation_index][0],operations[r_index][0],operations[m_index][0]]
                            generators_index = [srotation_index,r_index,m_index]



                    else:
                        if minues: # -nm     Dnd    odd n
                            if int(order_group/4) > 9:
                                group_symbol = f'-({int(order_group / 4)})m'
                                group_symbol_S = f'D{int(order_group / 4)}d'
                            else:
                                group_symbol = f'-{int(order_group/4)}m'
                                group_symbol_S = f'D{int(order_group/4)}d'
                            srotation_index, direction = find_rotation(operations,
                                                                      int(order_group / 4), np.array([0, 0, 1]),improper=True)
                            operations = reverse_direction(
                                operations,
                                direction)

                            m_index, m_direction = find_mirror(operations, np.array([1, 0, 0]),direction)
                            operations = reverse_direction(operations,m_direction)

                            generators = [operations[srotation_index][0],operations[m_index][0]]
                            generators_index = [srotation_index,m_index]
                        else: # -(2n)2m      Dnh   odd n
                            if int(order_group/2) > 9:
                                group_symbol = f'-({int(order_group / 2)})2m'
                                group_symbol_S = f'D{int(order_group / 4)}h'
                            else:
                                group_symbol = f'-{int(order_group/2)}2m'
                                group_symbol_S = f'D{int(order_group/4)}h'
                            srotation_index, direction = find_rotation(operations,
                                                                      int(order_group / 2), np.array([0, 0, 1]),improper=True)
                            operations = reverse_direction(
                                operations,
                                direction)
                            r_index, r_direction = find_rotation(operations,
                                                                      2, np.array([1, 0, 0]), direction)
                            operations = reverse_direction(operations,r_direction)
                            m_index, m_direction = find_mirror(operations, r_direction,direction,r_direction)
                            operations = reverse_direction(operations,m_direction)
                            generators = [operations[srotation_index][0],operations[r_index][0],operations[m_index][0]]
                            generators_index = [srotation_index,r_index,m_index]
        else: # Cn Dn
            if order_group == max_order: #   n     Cn
                if max_order > 9:
                    group_symbol = f'({max_order})'
                    group_symbol_S = f'C{max_order}'
                else:
                    group_symbol = f'{max_order}'
                    group_symbol_S = f'C{max_order}'
                rotation_index, direction = find_rotation(operations,
                                                           max_order, np.array([0, 0, 1]))
                operations = reverse_direction(
                    operations,
                    direction)
                generators = [operations[rotation_index][0]]  # matrix
                generators_index = [rotation_index]
            else: # Dn
                if int(order_group / 2) % 2 == 0: # n22   Dn
                    if max_order > 9:
                        group_symbol = f'({max_order})22'
                        group_symbol_S = f'D{max_order}'
                    else:
                        group_symbol = f'{max_order}22'
                        group_symbol_S = f'D{max_order}'
                    rotation_index, direction = find_rotation(operations,
                                                              max_order, np.array([0, 0, 1]))
                    operations = reverse_direction(
                        operations,
                        direction)

                    r_index, r_direction = find_rotation(operations,
                                                              2, np.array([1, 0, 0]),direction)
                    operations = reverse_direction(
                        operations,
                        r_direction)
                    r2_index, r2_direction = find_rotation(operations,
                                                       2, r_direction, direction, r_direction)
                    operations = reverse_direction(
                        operations,
                        r2_direction)
                    generators = [operations[rotation_index][0],operations[r_index][0],operations[r2_index][0]]  # matrix
                    generators_index = [rotation_index,r_index,r2_index]

                else:    # n2     Dn
                    if max_order > 9:
                        group_symbol = f'({max_order})2'
                        group_symbol_S = f'D{max_order}'
                    else:
                        group_symbol = f'{max_order}2'
                        group_symbol_S = f'D{max_order}'
                    rotation_index, direction = find_rotation(operations,
                                                              max_order, np.array([0, 0, 1]))
                    operations = reverse_direction(
                        operations,
                        direction)

                    r_index, r_direction = find_rotation(operations,
                                                              2, np.array([1, 0, 0]),direction)
                    operations = reverse_direction(
                        operations,
                        r_direction)
                    generators = [operations[rotation_index][0],
                                  operations[r_index][0]]  # matrix
                    generators_index = [rotation_index,r_index]
    if len(counter_high_order_axis) == 0: # Ci Cs C2 C2h C2v D2 D2h
        if order_group == 2:
            if minues: #Ci
                group_symbol = '-1'
                group_symbol_S = 'Ci'
                generators = [-np.eye(3)]
                generators_index = [np.where([i[1]==2 for i in operations] )[0][0]]
            elif improper: # Cs
                group_symbol = 'm'
                group_symbol_S = 'Cs'
                m_index, m_direction = find_mirror(operations, np.array([0, 1, 0]))
                operations = reverse_direction(
                    operations,
                    m_direction)
                generators = [operations[m_index][0]]
                generators_index = [m_index]
            else: # C2
                group_symbol = '2'
                group_symbol_S = 'C2'
                rotation_index, direction = find_rotation(operations,
                                                          2, np.array([0, 0, 1]))
                operations = reverse_direction(
                    operations,
                    direction)
                generators = [operations[rotation_index][0]]
                generators_index = [rotation_index]

        if order_group == 4:
            if minues:  # C2h
                group_symbol = '2/m'
                group_symbol_S = 'C2h'
                rotation_index, direction = find_rotation(operations,
                                                          2, np.array([0, 0, 1]))
                operations = reverse_direction(operations,
                                                                         direction)
                mirror_index, m_direction = find_mirror(operations, direction)

                generators = [operations[rotation_index][0],
                              operations[mirror_index][0]]  # matrix
                generators_index = [rotation_index,mirror_index]
            elif improper:   #C2v
                group_symbol = 'mm2'
                group_symbol_S = 'C2v'
                rotation_index, direction = find_rotation(operations,
                                                          2, np.array([0, 0, 1]))
                operations = reverse_direction(
                    operations,
                    direction)
                mirror_index, m_direction = find_mirror(operations, np.array([1, 0, 0]))
                operations = reverse_direction(operations,
                                                                         m_direction)
                mirror2_index, m2_direction = find_mirror(operations, m_direction,
                                                          exclude=m_direction)
                operations = reverse_direction(operations,
                                                                         m2_direction)
                generators = [
                              operations[mirror_index][0],
                              operations[mirror2_index][0],
                              operations[rotation_index][0]]
                generators_index = [mirror_index,mirror2_index,rotation_index]
            else:   # D2
                group_symbol = '222'
                group_symbol_S = 'D2'
                rotation_index, direction = find_rotation(operations,
                                                          2, np.array([0, 0, 1]))
                operations = reverse_direction(
                    operations,
                    direction)

                r_index, r_direction = find_rotation(operations,
                                                     2, np.array([1, 0, 0]), direction)
                operations = reverse_direction(
                    operations,
                    r_direction)
                r2_index, r2_direction = find_rotation(operations,
                                                       2, r_direction, direction, r_direction)
                operations = reverse_direction(
                    operations,
                    r2_direction)
                generators = [operations[rotation_index][0],
                              operations[r_index][0],
                              operations[r2_index][0]]  # matrix
                generators_index = [rotation_index,r_index,r2_index]

        if order_group == 8:    #D2h
            group_symbol = 'mmm'
            group_symbol_S = 'D2h'
            mz_index, mz_direction = find_mirror(operations, np.array([0, 0, 1]))
            m_index, m_direction = find_mirror(operations, np.array([1, 0, 0]), mz_direction)
            operations = reverse_direction(operations, m_direction)
            m2_index, m2_direction = find_mirror(operations, m_direction, mz_direction,
                                                 m_direction)
            operations = reverse_direction(operations, m2_direction)
            generators = [
                          operations[m_index][0],
                          operations[m2_index][0],
                          operations[mz_index][0]]
            generators_index = [m_index,m2_index,mz_index]

    # print(group_symbol)
    # print(generators_index)

    P2 = find_transition_matrix_deterministic(generators, group_symbol)
    transformation = P1 @ P2

    for op in operations:
        if op[3] is not None:
            op[0] = P1 @ op[0] @ P1_inv
            op[3] = P1 @ op[3]

    return group_symbol, operations, transformation, generators_index,group_symbol_S


def nonc_point_group_generators(n,index):
    if n < 2 or not isinstance(n,int) :
        raise TypeError('n must be an integer greater than or equal to 2')
    if  index < 0 or index > 9:
        raise ValueError('index must be an integer between 0 and 9')
    generators_lambda = [[lambda n: np.array([[math.cos(2*math.pi/n),-math.sin(2*math.pi/n),0],[math.sin(2*math.pi/n),math.cos(2*math.pi/n),0],[0,0,1]])],
                         [lambda n: np.array([[-math.cos(2*math.pi/n),math.sin(2*math.pi/n),0],[-math.sin(2*math.pi/n),-math.cos(2*math.pi/n),0],[0,0,-1]])],
                         [lambda n: np.array([[math.cos(2*math.pi/n),-math.sin(2*math.pi/n),0],[math.sin(2*math.pi/n),math.cos(2*math.pi/n),0],[0,0,1]]),lambda n: np.array([[1,0,0],[0,1,0],[0,0,-1]])],
                         [lambda n: np.array([[math.cos(2*math.pi/n),-math.sin(2*math.pi/n),0],[math.sin(2*math.pi/n),math.cos(2*math.pi/n),0],[0,0,1]]),lambda n: np.array([[-1,0,0],[0,1,0],[0,0,1]]),lambda n: np.array([[math.cos(2*math.pi/n),-math.sin(2*math.pi/n),0],[math.sin(2*math.pi/n),math.cos(2*math.pi/n),0],[0,0,1]]) @np.array([[-1,0,0],[0,1,0],[0,0,1]])],
                         [lambda n: np.array([[math.cos(2*math.pi/n),-math.sin(2*math.pi/n),0],[math.sin(2*math.pi/n),math.cos(2*math.pi/n),0],[0,0,1]]),lambda n: np.array([[-1,0,0],[0,1,0],[0,0,1]])],
                         [lambda n: np.array([[math.cos(2*math.pi/n),-math.sin(2*math.pi/n),0],[math.sin(2*math.pi/n),math.cos(2*math.pi/n),0],[0,0,1]]),lambda n: np.array([[1,0,0],[0,1,0],[0,0,-1]]),lambda n: np.array([[-1,0,0],[0,1,0],[0,0,1]]),lambda n: np.array([[math.cos(2*math.pi/n),-math.sin(2*math.pi/n),0],[math.sin(2*math.pi/n),math.cos(2*math.pi/n),0],[0,0,1]]) @np.array([[-1,0,0],[0,1,0],[0,0,1]])],
                         [lambda n: np.array([[-math.cos(2*math.pi/n),math.sin(2*math.pi/n),0],[-math.sin(2*math.pi/n),-math.cos(2*math.pi/n),0],[0,0,-1]]),lambda n: np.array([[1,0,0],[0,-1,0],[0,0,-1]]),lambda n: np.array([[-math.cos(2*math.pi/n),math.sin(2*math.pi/n),0],[-math.sin(2*math.pi/n),-math.cos(2*math.pi/n),0],[0,0,-1]]) @np.array([[1,0,0],[0,-1,0],[0,0,-1]]) ],
                         [lambda n: np.array([[-math.cos(2*math.pi/n),math.sin(2*math.pi/n),0],[-math.sin(2*math.pi/n),-math.cos(2*math.pi/n),0],[0,0,-1]]),lambda n: np.array([[-1,0,0],[0,1,0],[0,0,1]])],
                         [lambda n: np.array([[math.cos(2*math.pi/n),-math.sin(2*math.pi/n),0],[math.sin(2*math.pi/n),math.cos(2*math.pi/n),0],[0,0,1]]),lambda n: np.array([[1,0,0],[0,-1,0],[0,0,-1]]),lambda n: np.array([[math.cos(2*math.pi/n),-math.sin(2*math.pi/n),0],[math.sin(2*math.pi/n),math.cos(2*math.pi/n),0],[0,0,1]])@np.array([[1,0,0],[0,-1,0],[0,0,-1]])],
                         [lambda n: np.array([[math.cos(2*math.pi/n),-math.sin(2*math.pi/n),0],[math.sin(2*math.pi/n),math.cos(2*math.pi/n),0],[0,0,1]]),lambda n: np.array([[1,0,0],[0,-1,0],[0,0,-1]])]
                         ]
    return [i(n) for i in generators_lambda[index]]


def generate_non_crystallographic_point_groups(hm_symbol):
    """
        input hm_symbol
        n  -n  n/m  nmm  nm  n/mmm  -n2m  -nm  n22  n2


        return generators(in cartesian system)


    """


    if bool(re.search(r'\(', hm_symbol)):
        patterns = [r'\((\d+)\)', r'-(\d+)', r'(\d+)/m', r'(\d+)mm', r"(\d+)m", r'(\d+)/mmm', r'-\((\d+)\)2m', r'-\((\d+)\)m', r'\((\d+)\)22',
                r'\((\d+)\)2']

    else:
        patterns = [r'(\d)', r'-(\d)', r'(\d+)/m', r'(\d+)mm', r"(\d+)m", r'(\d+)/mmm', r'-(\d)2m', r'-(\d)m',
                    r'(\d)22',r'(\d)2']
    patterns = [re.compile(p) for p in patterns]

    n = None
    pattern_index = None
    for index,pattern in enumerate(patterns):
        match = re.fullmatch(pattern, hm_symbol)
        if match:
            n = int(match.group(1))
            pattern_index = index
    if n is None:
        raise ValueError('No pattern found')
    generators = nonc_point_group_generators(n,pattern_index)

    return generators






def find_transition_matrix_deterministic(matrices_list, group_symbol, tol=1e-2):
    """
    确定性地计算非奇异转换矩阵 P。

    修正点：
    1. 严格保持与原 random 函数一致的 reshape 和 return P.T 逻辑。
    2. 使用更稳健的基向量混合策略，确保 det(P) 不为 0。
    """
    from findspingroup.data.POINT_GROUP_MATRIX import point_group_generators

    # 1. 准备标准生成元
    if group_symbol not in point_group_generators:
        standard_list = generate_non_crystallographic_point_groups(group_symbol)
        # raise ValueError(f"不支持的点群: {group_symbol}")
    else:
        standard_list = [np.array(op) for op in point_group_generators[group_symbol]]

    if len(matrices_list) != len(standard_list):
        raise ValueError(f"矩阵数量 ({len(matrices_list)}) 与标准生成元数量 ({len(standard_list)}) 不匹配。")

    # 2. 构建线性方程组
    # 逻辑维持原样：对应的 kron 构造需要配合最后的 return P.T
    I = np.eye(3)
    A_blocks = []
    for M, D in zip(matrices_list, standard_list):
        A_block = np.kron(I, M) - np.kron(D.T, I)
        A_blocks.append(A_block)

    A = np.vstack(A_blocks)

    # 3. SVD分解求零空间
    U, S, Vh = np.linalg.svd(A)

    # 获取零空间基向量 (对应于极小的奇异值)
    null_mask = S < tol
    null_space_dim = np.sum(null_mask)

    # 如果 SVD 数值误差导致没有捕捉到零空间 (rank deficient)，强制取最后几行
    if null_space_dim == 0:
        # 强行尝试取最小的奇异值对应的向量，或者报错
        # 通常物理上必定有解，这里给一个保底
        if S[-1] < 0.1:
            null_space_dim = 1
        else:
            raise ValueError("没有找到零空间，输入矩阵可能不构成指定的点群。")

    # 取 Vh 的最后几行作为基向量
    basis_vectors_flat = Vh[-null_space_dim:, :]

    # 将基向量 reshape 为 3x3 矩阵
    # 注意：这里不做 .T，完全模拟原代码 P_flat.reshape(3,3) 的行为
    basis_matrices = [b.reshape(3, 3) for b in basis_vectors_flat]

    # 4. 确定性混合基向量寻找非奇异解
    # 策略：贪婪叠加。
    # P_final = v1 + c2*v2 + c3*v3 ...

    current_P = basis_matrices[0]

    # 如果零空间维度 > 1，我们需要混合它们以避开奇异点
    if len(basis_matrices) > 1:
        for i, V in enumerate(basis_matrices[1:]):
            # 如果当前已经是非奇异且行列式较大，可以直接停止，也可以继续混合增加鲁棒性
            # 这里设定一个满意的阈值
            if abs(np.linalg.det(current_P)) > 0.5:
                break

            # 尝试不同的系数混合，寻找能显著增加行列式的方向
            best_alpha = 0
            max_det = abs(np.linalg.det(current_P))

            # 扫描系数：这保证了如果在这个方向上有非奇异解，我们一定能撞上
            # 使用素数或非整数步长可以避免巧合的整数抵消
            coeffs = [1.0, -1.0, 0.5, 2.0, 1.5, -1.5]

            for alpha in coeffs:
                temp_P = current_P + alpha * V
                det_val = abs(np.linalg.det(temp_P))
                if det_val > max_det:
                    max_det = det_val
                    best_alpha = alpha

            if best_alpha != 0:
                current_P = current_P + best_alpha * V

    # 5. 检查结果
    if abs(np.linalg.det(current_P)) < 1e-4:
        # 极少数情况：如果确定性搜索失败（非常罕见），
        # 可以在这个狭窄的零空间内做最后一次随机尝试作为保底
        # print("Deterministic mix failed, trying random mix in null space...")
        for _ in range(50):
            coeffs = np.random.uniform(-2, 2, size=len(basis_matrices))
            P_rand = sum(c * B for c, B in zip(coeffs, basis_matrices))
            if abs(np.linalg.det(P_rand)) > 1e-3:
                current_P = P_rand
                break

        if abs(np.linalg.det(current_P)) < 1e-4:
            raise ValueError("无法在零空间中找到非奇异矩阵 P。")

    # 6. 返回结果
    # 必须保留 .T，因为原代码的 Kronecker 构建方式实际上解出的是 P^T
    return current_P.T
def find_transition_matrix_random(matrices_list, group_symbol, tol=1e-2, max_tries=1000):
    """
    随机尝试，直到找到一个非奇异的 P。

    输入:
      matrices_list: list of 3x3 numpy.ndarray
      group_symbol: str，比如 "C3"

    输出:
      P: 3x3 numpy.ndarray
    """

    from findspingroup.data.POINT_GROUP_MATRIX import point_group_generators

    if group_symbol not in point_group_generators: #
        standard_list = generate_non_crystallographic_point_groups(group_symbol)
        # raise ValueError(f"不支持的点群: {group_symbol}")
    else:
        standard_list = [np.array(op) for op in point_group_generators[group_symbol]]

    if len(matrices_list) != len(standard_list):
        raise ValueError(f"矩阵数量 ({len(matrices_list)}) 与标准生成元数量 ({len(standard_list)}) 不匹配。")

    I = np.eye(3)
    A_blocks = []
    for M, D in zip(matrices_list, standard_list):
        A_block = np.kron(I, M) - np.kron(D.T, I)
        A_blocks.append(A_block)

    A = np.vstack(A_blocks)

    # SVD分解
    U, S, Vh = np.linalg.svd(A)
    null_space_dim = np.sum(S < tol)
    if null_space_dim == 0:
        raise ValueError("没有零空间，可能群元素给错了。")

    basis_vectors = Vh[-null_space_dim:, :]  # nullspace basis

    tries = 0
    while tries < max_tries:
        random_coeffs = np.random.randint(-5, 5, size=null_space_dim)
        P_flat = random_coeffs @ basis_vectors
        P = P_flat.reshape(3, 3)

        if abs(np.linalg.det(P)) > 0.3:
            #

            # print(f"Found non-singular P in {tries + 1} tries.")
            return P.T

        tries += 1

    raise ValueError(f"在 {max_tries} 次尝试中未找到非奇异的P。")


def getNormInf(matrix1, matrix2, mode=True):
    if mode == True:
        a = np.array(matrix1) % 1
        b = np.array(matrix2) % 1
        c = [1, 2, 3]
        for i in range(3):
            if a[i] > b[i]:
                c[i] = min(a[i] - b[i], 1 + b[i] - a[i])
            if a[i] < b[i]:
                c[i] = min(b[i] - a[i], 1 + a[i] - b[i])
            if a[i] == b[i]:
                c[i] = 0
        max_value = max(c)
    else:
        diff = np.abs(matrix1 - matrix2)
        max_value = np.max(diff)
    return max_value


def is_close_matrix_pair(pair1, pair2, tol=1e-5):
    if len(pair1) != len(pair2):
        raise ValueError("Compare two vectors of different lengths.")
    for i, j in enumerate(pair1):
        if not np.allclose(np.array(pair1[i]), np.array(pair2[i]), atol=tol):
            return False
    return True


def deduplicate_matrix_pairs(matrix_list, tol=1e-5):
    unique = []
    for item in matrix_list:
        if not any(is_close_matrix_pair(item, u, tol) for u in unique):
            unique.append(item)
    return unique


def compute_invariant_metric(point_group_rotations):
    """calculate invariant metric tensor g。"""
    if not point_group_rotations:
        raise ValueError("no point group rotations provided.")

    g0 = np.eye(3)
    g = np.zeros((3, 3))
    for R in point_group_rotations:
        g += np.dot(R.T, np.dot(g0, R))
    g /= len(point_group_rotations)

    # check positive definiteness
    eigvals = np.linalg.eigvalsh(g)
    if np.min(eigvals) < 1e-6:
        raise ValueError("cannot compute invariant metric tensor, not positive definite.")

    return g


def get_space_group_from_operations(space_group_operations,symprec = 0.02,bz = False)->SpglibDataset:


    weird_sites = [np.array([0.1715870, 0.27754210, 0.737388700]),np.array([0,0,0])]

    # get point group rotations
    point_group_rotations = deduplicate_matrix_pairs([i[0] for i in space_group_operations])
    g = compute_invariant_metric(point_group_rotations)
    lattice = np.linalg.cholesky(g)  # L @ L.T = g , rows as basis vectors

    positions = []
    types = []
    for index,site in enumerate(weird_sites):
        for op in space_group_operations:
            new_pos = np.dot(op[0], site) + op[1]
            new_pos = np.mod(new_pos, 1)
            if not any(getNormInf(new_pos,p) < 1e-3 for p in positions):
                positions.append(new_pos)
                types.append(index+1)

    cell = (lattice*20, positions, types)

    space_group_dataset =get_symmetry_dataset(cell, symprec=symprec)
    if space_group_dataset.number in SG_HALL_MAPPING:
        # corresponding to the same space group setting as in Bilbao Crystallographic Server
        space_group_dataset =get_symmetry_dataset(cell, symprec=symprec, hall_number=SG_HALL_MAPPING[space_group_dataset.number])

    if bz :
        path_info = get_path(cell,with_time_reversal=False,symprec=symprec)
        return  space_group_dataset, path_info
    else:
        return space_group_dataset


def get_magnetic_space_group_from_operations(magnetic_space_group_operations):
    """
    :param magnetic_space_group_operations: [[ time_reversal{1,-1},rotation, translation)],...]
    :return : dict with keys:
        msg_int_num: int, magnetic space group international number
        msg_bns_num: str, magnetic space group BNS number
        msg_bns_symbol: str, magnetic space group BNS symbol
        msg_og_num: int, magnetic space group OG number
        msg_og_symbol: str, magnetic space group OG symbol
        msg_type: int, magnetic space group type (1-4)
        mpg_num: int, magnetic point group number
        mpg_symbol: str, magnetic point group symbol
    """
    symprec = 0.02

    weird_sites = [np.array([0.1715870, 0.27754210, 0.737388700]),np.array([0,0,0])]
    weird_moments = [np.array([1.234,0.789,0.345]),np.array([0,0,0])]

    # get point group rotations
    point_group_rotations = deduplicate_matrix_pairs([i[1] for i in magnetic_space_group_operations])
    g = compute_invariant_metric(point_group_rotations)
    lattice = np.linalg.cholesky(g)  # L @ L.T = g , rows as basis vectors

    positions = []
    types = []
    moments = []
    for index,site in enumerate(weird_sites):
        for op in magnetic_space_group_operations:
            new_pos = np.dot(op[1], site) + op[2]
            new_pos = np.mod(new_pos, 1)
            if not any(getNormInf(new_pos,p) < 1e-4 for i,p in enumerate(positions)):
                positions.append(new_pos)
                types.append(index+1)
                new_mom = np.dot(round(np.linalg.det(op[1])) * op[1] * op[0], weird_moments[index])
                moments.append(new_mom)

    cell = (lattice, positions, types, moments@lattice)
    magnetic_space_group_dataset :SpglibMagneticDataset=get_magnetic_symmetry_dataset(cell, symprec=symprec,mag_symprec=0.02)

    msg_int_num = magnetic_space_group_dataset.uni_number
    msg_bns_num,msg_bns_symbol = MSG_INT_TO_BNS[msg_int_num]
    msg_og_num = BNS_TO_OG_NUM[msg_bns_num]
    msg_og_symbol = OG_NUM_TO_MPG[msg_og_num]["og_label"]
    msg_type = magnetic_space_group_dataset.msg_type
    mpg_num = OG_NUM_TO_MPG[msg_og_num]["pointgroup_no"]
    mpg_symbol = OG_NUM_TO_MPG[msg_og_num]["pointgroup_label"]



    return {"msg_int_num":msg_int_num,
            "msg_bns_num":msg_bns_num,
            "msg_bns_symbol":msg_bns_symbol,
            "msg_og_num":msg_og_num,
            "msg_og_symbol":msg_og_symbol,
            "msg_type":msg_type,
            "mpg_num":mpg_num,
            "mpg_symbol":mpg_symbol}


def get_arithmetic_crystal_class_from_ops(ops):
    """
    rely on spglib
    :parameter: ops: list of [rotation matrix, translation vector], space group operations
    :return: acc_symbol: str, arithmetic crystal class symbol
    """


    # get point group rotations
    acc_dataset,kpath_info = get_space_group_from_operations(ops,bz=True)

    if acc_dataset is None:
        raise ValueError("Can not find spg dataset in arithmetic crystal class ")

    international = acc_dataset.international
    bravais_lattice_letter = acc_dataset.international[0]
    pointgroup = acc_dataset.pointgroup

    # process 66 -> 73 TODO: test mm2C and mm2A
    if pointgroup == '-42m':
        if international[1:4] =='-42':
            pointgroup = '-42m'
        else:
            pointgroup = '-4m2'
    if pointgroup == '32' and bravais_lattice_letter == 'P':
        if international[-2:] == '12':
            pointgroup = '312'
        else:
            pointgroup = '321'

    if pointgroup == '3m' and bravais_lattice_letter == 'P':
        if international[-1] == '1':
            pointgroup = '3m1'
        else:
            pointgroup = '31m'

    if pointgroup == '-3m' and bravais_lattice_letter == 'P':
        if international[-1] == '1':
            pointgroup = '-3m1'
        else:
            pointgroup = '-31m'

    if pointgroup == '-62m':
        if international[-1] == '2':
            pointgroup = '-6m2'
        else:
            pointgroup = '-62m'

    acc_symbol = pointgroup + bravais_lattice_letter
    from ..structure.cell import primitive_cell_transformation


    input_acc_transformation = np.linalg.inv(primitive_cell_transformation(acc_dataset.international)) @ acc_dataset.transformation_matrix
    input_acc_origin_shift = normalize_vector_to_zero(np.linalg.inv(primitive_cell_transformation(acc_dataset.international)) @ acc_dataset.origin_shift ,atol=1e-9)
    # L_input = L_acc_std @ input_acc_std_transformation , L for col vector
    # L_acc = L_acc_std @ primitive_transformation
    # L_input = L_acc @ primitive_transformation^-1 @ input_acc_std_transformation
    # L_input = L_acc @ input_acc_transformation
    return acc_symbol, input_acc_transformation,input_acc_origin_shift,kpath_info

