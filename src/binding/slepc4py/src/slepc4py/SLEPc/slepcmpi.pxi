# -----------------------------------------------------------------------------

cdef extern from * nogil:
    const MPI_Comm MPI_COMM_NULL
    const MPI_Comm MPI_COMM_SELF
    const MPI_Comm MPI_COMM_WORLD

cdef extern from * nogil:
    MPI_Comm PETSC_COMM_SELF
    MPI_Comm PETSC_COMM_WORLD

# -----------------------------------------------------------------------------

from petsc4py.PETSc cimport GetComm
cdef inline MPI_Comm def_Comm(
    object comm,
    MPI_Comm defv,
) except? MPI_COMM_NULL:
    return GetComm(comm, defv)

from petsc4py.PETSc cimport GetCommDefault
cdef inline MPI_Comm SLEPC_COMM_DEFAULT() except? MPI_COMM_NULL:
    return GetCommDefault()

# -----------------------------------------------------------------------------
