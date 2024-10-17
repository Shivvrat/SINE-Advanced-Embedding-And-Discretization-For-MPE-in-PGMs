# distutils: language = c++
# cython: language_level=3

import numpy as np
cimport numpy as np
from libcpp.vector cimport vector
from libcpp.algorithm cimport sort
from libcpp.utility cimport pair

np.import_array()

ctypedef pair[double, vector[int]] Assignment

cdef inline int compare_assignments(const Assignment& a, const Assignment& b) nogil:
    return a.first < b.first

cdef vector[Assignment] process_assignments(const double* s_i_ptr, int N, int k) nogil:
    cdef int i, idx
    cdef double s_idx, D, D_new_0, D_new_1
    cdef vector[Assignment] L, T
    cdef Assignment current_assignment, new_assignment_0, new_assignment_1

    L.push_back(Assignment(0.0, vector[int]()))

    for idx in range(N):
        T.clear()
        s_idx = s_i_ptr[idx]

        for i in range(L.size()):
            current_assignment = L[i]
            D = current_assignment.first

            # Case 1: b_i = 0
            D_new_0 = D + s_idx
            new_assignment_0 = Assignment(D_new_0, current_assignment.second)
            new_assignment_0.second.push_back(0)

            # Case 2: b_i = 1
            D_new_1 = D + (1 - s_idx)
            new_assignment_1 = Assignment(D_new_1, current_assignment.second)
            new_assignment_1.second.push_back(1)

            T.push_back(new_assignment_0)
            T.push_back(new_assignment_1)

        sort(T.begin(), T.end(), compare_assignments)
        
        if T.size() > k:
            T.resize(k)
        
        L = T

    return L

def cython_process_assignments(np.ndarray[np.float64_t, ndim=1] s_i, int k):
    cdef int N = s_i.shape[0]
    cdef vector[Assignment] result = process_assignments(&s_i[0], N, k)
    
    # Convert the result to Python lists
    cdef list py_result = [(assignment.first, assignment.second) for assignment in result]
    
    # Create a NumPy array for assignments
    cdef np.ndarray[np.int32_t, ndim=2] assignments = np.zeros((len(py_result), N), dtype=np.int32)
    for i, (_, assignment) in enumerate(py_result):
        for j, value in enumerate(assignment):
            assignments[i, j] = value
    
    return py_result, assignments