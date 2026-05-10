"""Generator-only spin-space-group operation cases for spintensor tests.

Each case was reduced from the finite Rs/Rl pairs in an OUTPUT file2.txt source.
Translations and continuous spin-only phi operations are intentionally omitted.
"""

SPINTENSOR_GENERATOR_CASES = [
    {
        "name": "collinear_typeI_135_135_1_1_L",
        "source": "OUTPUT/Collinear/typeI/135/135/135.135.1.1.L/file2.txt",
        "ops": [
            {
                "name": "collinear_typeI_135_135_1_1_L__g02_1_real_41_001",
                "Rs": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                "Rl": [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
            },
            {
                "name": "collinear_typeI_135_135_1_1_L__g03_1_real_m_001",
                "Rs": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                "Rl": [[1, 0, 0], [0, 1, 0], [0, 0, -1]],
            },
            {
                "name": "collinear_typeI_135_135_1_1_L__g04_1_real_m_100",
                "Rs": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                "Rl": [[-1, 0, 0], [0, 1, 0], [0, 0, 1]],
            },
            {
                "name": "collinear_typeI_135_135_1_1_L__g03_m_100_spin",
                "Rs": [[-1, 0, 0], [0, 1, 0], [0, 0, 1]],
                "Rl": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            },
        ],
    },
    {
        "name": "collinear_typeI_166_166_1_1_L",
        "source": "OUTPUT/Collinear/typeI/166/166/166.166.1.1.L/file2.txt",
        "ops": [
            {
                "name": "collinear_typeI_166_166_1_1_L__g02_1_real_m31_001",
                "Rs": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                "Rl": [[0, 1, 0], [-1, 1, 0], [0, 0, -1]],
            },
            {
                "name": "collinear_typeI_166_166_1_1_L__g03_1_real_m_100",
                "Rs": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                "Rl": [[-1, 1, 0], [0, 1, 0], [0, 0, 1]],
            },
            {
                "name": "collinear_typeI_166_166_1_1_L__g03_m_100_spin",
                "Rs": [[-1, 0, 0], [0, 1, 0], [0, 0, 1]],
                "Rl": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            },
        ],
    },
    {
        "name": "collinear_typeI_229_229_1_1_L",
        "source": "OUTPUT/Collinear/typeI/229/229/229.229.1.1.L/file2.txt",
        "ops": [
            {
                "name": "collinear_typeI_229_229_1_1_L__g03_1_real_m31_111",
                "Rs": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                "Rl": [[0, 0, -1], [-1, 0, 0], [0, -1, 0]],
            },
            {
                "name": "collinear_typeI_229_229_1_1_L__g04_1_real_m_110",
                "Rs": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                "Rl": [[0, -1, 0], [-1, 0, 0], [0, 0, 1]],
            },
            {
                "name": "collinear_typeI_229_229_1_1_L__g03_m_100_spin",
                "Rs": [[-1, 0, 0], [0, 1, 0], [0, 0, 1]],
                "Rl": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            },
        ],
    },
    {
        "name": "collinear_typeII_59_59_2_1_L",
        "source": "OUTPUT/Collinear/typeII/59/59/59.59.2.1.L/file2.txt",
        "ops": [
            {
                "name": "collinear_typeII_59_59_2_1_L__g02_1_real_m_100",
                "Rs": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                "Rl": [[-1, 0, 0], [0, 1, 0], [0, 0, 1]],
            },
            {
                "name": "collinear_typeII_59_59_2_1_L__g03_1_real_m_010",
                "Rs": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                "Rl": [[1, 0, 0], [0, -1, 0], [0, 0, 1]],
            },
            {
                "name": "collinear_typeII_59_59_2_1_L__g04_1_real_m_001",
                "Rs": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                "Rl": [[1, 0, 0], [0, 1, 0], [0, 0, -1]],
            },
            {
                "name": "collinear_typeII_59_59_2_1_L__g03_m_100_spin",
                "Rs": [[-1, 0, 0], [0, 1, 0], [0, 0, 1]],
                "Rl": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            },
        ],
    },
    {
        "name": "collinear_typeII_168_168_2_1_L",
        "source": "OUTPUT/Collinear/typeII/168/168/168.168.2.1.L/file2.txt",
        "ops": [
            {
                "name": "collinear_typeII_168_168_2_1_L__g02_1_real_61_001",
                "Rs": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                "Rl": [[1, -1, 0], [1, 0, 0], [0, 0, 1]],
            },
            {
                "name": "collinear_typeII_168_168_2_1_L__g03_m_100_spin",
                "Rs": [[-1, 0, 0], [0, 1, 0], [0, 0, 1]],
                "Rl": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            },
        ],
    },
    {
        "name": "coplanar_typeI_135_32_1_1_P1",
        "source": "OUTPUT/Coplanar/typeI/135/32/135.32.1.1.P1/file2.txt",
        "ops": [
            {
                "name": "coplanar_typeI_135_32_1_1_P1__g02_2_001_real_41_001",
                "Rs": [[-1, 0, 0], [0, -1, 0], [0, 0, 1]],
                "Rl": [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
            },
            {
                "name": "coplanar_typeI_135_32_1_1_P1__g03_2_010_real_m_001",
                "Rs": [[-1, 0, 0], [0, 1, 0], [0, 0, -1]],
                "Rl": [[1, 0, 0], [0, 1, 0], [0, 0, -1]],
            },
            {
                "name": "coplanar_typeI_135_32_1_1_P1__g04_1_real_m_100",
                "Rs": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                "Rl": [[-1, 0, 0], [0, 1, 0], [0, 0, 1]],
            },
            {
                "name": "coplanar_typeI_135_32_1_1_P1__g02_m_001_spin",
                "Rs": [[1, 0, 0], [0, 1, 0], [0, 0, -1]],
                "Rl": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            },
        ],
    },
    {
        "name": "coplanar_typeI_135_6_1_1_P",
        "source": "OUTPUT/Coplanar/typeI/135/6/135.6.1.1.P/file2.txt",
        "ops": [
            {
                "name": "coplanar_typeI_135_6_1_1_P__g02_41_001_real_41_001",
                "Rs": [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
                "Rl": [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
            },
            {
                "name": "coplanar_typeI_135_6_1_1_P__g03_1_real_m_001",
                "Rs": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                "Rl": [[1, 0, 0], [0, 1, 0], [0, 0, -1]],
            },
            {
                "name": "coplanar_typeI_135_6_1_1_P__g04_2_1m10_real_m_100",
                "Rs": [[0, -1, 0], [-1, 0, 0], [0, 0, -1]],
                "Rl": [[-1, 0, 0], [0, 1, 0], [0, 0, 1]],
            },
            {
                "name": "coplanar_typeI_135_6_1_1_P__g02_m_001_spin",
                "Rs": [[1, 0, 0], [0, 1, 0], [0, 0, -1]],
                "Rl": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            },
        ],
    },
    {
        "name": "coplanar_typeII_95_95_2_2_P1",
        "source": "OUTPUT/Coplanar/typeII/95/95/95.95.2.2.P1/file2.txt",
        "ops": [
            {
                "name": "coplanar_typeII_95_95_2_2_P1__g02_1_real_41_001",
                "Rs": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                "Rl": [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
            },
            {
                "name": "coplanar_typeII_95_95_2_2_P1__g03_1_real_2_100",
                "Rs": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                "Rl": [[1, 0, 0], [0, -1, 0], [0, 0, -1]],
            },
            {
                "name": "coplanar_typeII_95_95_2_2_P1__g02_m_001_spin",
                "Rs": [[1, 0, 0], [0, 1, 0], [0, 0, -1]],
                "Rl": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            },
        ],
    },
    {
        "name": "coplanar_typeII_132_137_2_2_P2",
        "source": "OUTPUT/Coplanar/typeII/132/137/132.137.2.2.P2/file2.txt",
        "ops": [
            {
                "name": "coplanar_typeII_132_137_2_2_P2__g02_1_real_41_001",
                "Rs": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                "Rl": [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
            },
            {
                "name": "coplanar_typeII_132_137_2_2_P2__g03_1_real_m_001",
                "Rs": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                "Rl": [[1, 0, 0], [0, 1, 0], [0, 0, -1]],
            },
            {
                "name": "coplanar_typeII_132_137_2_2_P2__g04_1_real_m_100",
                "Rs": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                "Rl": [[-1, 0, 0], [0, 1, 0], [0, 0, 1]],
            },
            {
                "name": "coplanar_typeII_132_137_2_2_P2__g02_m_100_spin",
                "Rs": [[-1, 0, 0], [0, 1, 0], [0, 0, 1]],
                "Rl": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            },
        ],
    },
    {
        "name": "coplanar_typeIII_135_32_6_4_P",
        "source": "OUTPUT/Coplanar/typeIII/135/32/135.32.6.4.P/file2.txt",
        "ops": [
            {
                "name": "coplanar_typeIII_135_32_6_4_P__g02_121_001_real_41_001",
                "Rs": [
                    [0.866025403784, -0.5, 0],
                    [0.5, 0.866025403784, 0],
                    [0, 0, 1],
                ],
                "Rl": [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
            },
            {
                "name": "coplanar_typeIII_135_32_6_4_P__g03_2_100_real_m_001",
                "Rs": [[1, 0, 0], [0, -1, 0], [0, 0, -1]],
                "Rl": [[1, 0, 0], [0, 1, 0], [0, 0, -1]],
            },
            {
                "name": "coplanar_typeIII_135_32_6_4_P__g04_1_real_m_100",
                "Rs": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                "Rl": [[-1, 0, 0], [0, 1, 0], [0, 0, 1]],
            },
            {
                "name": "coplanar_typeIII_135_32_6_4_P__g02_m_001_spin",
                "Rs": [[1, 0, 0], [0, 1, 0], [0, 0, -1]],
                "Rl": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            },
        ],
    },
    {
        "name": "coplanar_typeIII_135_56_2_1_P",
        "source": "OUTPUT/Coplanar/typeIII/135/56/135.56.2.1.P/file2.txt",
        "ops": [
            {
                "name": "coplanar_typeIII_135_56_2_1_P__g02_41_001_real_41_001",
                "Rs": [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
                "Rl": [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
            },
            {
                "name": "coplanar_typeIII_135_56_2_1_P__g03_2_001_real_m_001",
                "Rs": [[-1, 0, 0], [0, -1, 0], [0, 0, 1]],
                "Rl": [[1, 0, 0], [0, 1, 0], [0, 0, -1]],
            },
            {
                "name": "coplanar_typeIII_135_56_2_1_P__g04_41_001_real_m_100",
                "Rs": [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
                "Rl": [[-1, 0, 0], [0, 1, 0], [0, 0, 1]],
            },
            {
                "name": "coplanar_typeIII_135_56_2_1_P__g02_m_001_spin",
                "Rs": [[1, 0, 0], [0, 1, 0], [0, 0, -1]],
                "Rl": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            },
        ],
    },
    {
        "name": "noncoplanar_typeI_135_32_1_5",
        "source": "OUTPUT/Non-coplanar/typeI/135/32/135.32.1.5/file2.txt",
        "ops": [
            {
                "name": "noncoplanar_typeI_135_32_1_5__g02_2_001_real_41_001",
                "Rs": [[-1, 0, 0], [0, -1, 0], [0, 0, 1]],
                "Rl": [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
            },
            {
                "name": "noncoplanar_typeI_135_32_1_5__g03_m1_real_m_001",
                "Rs": [[-1, 0, 0], [0, -1, 0], [0, 0, -1]],
                "Rl": [[1, 0, 0], [0, 1, 0], [0, 0, -1]],
            },
            {
                "name": "noncoplanar_typeI_135_32_1_5__g04_1_real_m_100",
                "Rs": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                "Rl": [[-1, 0, 0], [0, 1, 0], [0, 0, 1]],
            },
        ],
    },
    {
        "name": "noncoplanar_typeI_166_1_1_6",
        "source": "OUTPUT/Non-coplanar/typeI/166/1/166.1.1.6/file2.txt",
        "ops": [
            {
                "name": "noncoplanar_typeI_166_1_1_6__g02_m61_001_real_m31_001",
                "Rs": [
                    [-0.5, 0.866025403784, 0],
                    [-0.866025403784, -0.5, 0],
                    [0, 0, -1],
                ],
                "Rl": [[0, 1, 0], [-1, 1, 0], [0, 0, -1]],
            },
            {
                "name": "noncoplanar_typeI_166_1_1_6__g03_2_2pi_3_real_m_100",
                "Rs": [
                    [-0.5, -0.866025403784, 0],
                    [-0.866025403784, 0.5, 0],
                    [0, 0, -1],
                ],
                "Rl": [[-1, 1, 0], [0, 1, 0], [0, 0, 1]],
            },
        ],
    },
    {
        "name": "noncoplanar_typeII_95_95_2_3",
        "source": "OUTPUT/Non-coplanar/typeII/95/95/95.95.2.3/file2.txt",
        "ops": [
            {
                "name": "noncoplanar_typeII_95_95_2_3__g02_1_real_41_001",
                "Rs": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                "Rl": [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
            },
            {
                "name": "noncoplanar_typeII_95_95_2_3__g03_1_real_2_100",
                "Rs": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                "Rl": [[1, 0, 0], [0, -1, 0], [0, 0, -1]],
            },
        ],
    },
    {
        "name": "noncoplanar_typeIII_135_32_3_6",
        "source": "OUTPUT/Non-coplanar/typeIII/135/32/135.32.3.6/file2.txt",
        "ops": [
            {
                "name": "noncoplanar_typeIII_135_32_3_6__g02_m61_001_real_41_001",
                "Rs": [
                    [-0.5, 0.866025403784, 0],
                    [-0.866025403784, -0.5, 0],
                    [0, 0, -1],
                ],
                "Rl": [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
            },
            {
                "name": "noncoplanar_typeIII_135_32_3_6__g03_m_5pi_6_real_m_001",
                "Rs": [
                    [-0.5, 0.866025403784, 0],
                    [0.866025403784, 0.5, 0],
                    [0, 0, 1],
                ],
                "Rl": [[1, 0, 0], [0, 1, 0], [0, 0, -1]],
            },
            {
                "name": "noncoplanar_typeIII_135_32_3_6__g04_1_real_m_100",
                "Rs": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                "Rl": [[-1, 0, 0], [0, 1, 0], [0, 0, 1]],
            },
        ],
    },
]
