import numpy as np
from scipy.sparse import csr_matrix ,load_npz
A = load_npz('combined_efficiency_matrix.npz')
B = load_npz('system_matrix_voxel_source.npz')
print(A.shape)
print(B.shape)