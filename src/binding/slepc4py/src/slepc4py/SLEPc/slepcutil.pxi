cdef extern from * nogil:
    PetscErrorCode MatCreateBSE(PetscMat,PetscMat,PetscMat*)

