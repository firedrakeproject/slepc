cdef extern from * nogil:

    ctypedef char* SlepcMFNType "const char*"
    SlepcMFNType MFNKRYLOV
    SlepcMFNType MFNEXPOKIT

    ctypedef enum SlepcMFNConvergedReason "MFNConvergedReason":
        MFN_CONVERGED_TOL
        MFN_CONVERGED_ITS
        MFN_DIVERGED_ITS
        MFN_DIVERGED_BREAKDOWN
        MFN_CONVERGED_ITERATING

    ctypedef int (*SlepcMFNCtxDel)(void*)
    ctypedef int (*SlepcMFNMonitorFunction)(SlepcMFN,
                                            PetscInt,
                                            PetscReal,
                                            void*) except PETSC_ERR_PYTHON

    int MFNCreate(MPI_Comm,SlepcMFN*)
    int MFNDestroy(SlepcMFN*)
    int MFNReset(SlepcMFN)
    int MFNView(SlepcMFN,PetscViewer)

    int MFNSetType(SlepcMFN,SlepcMFNType)
    int MFNGetType(SlepcMFN,SlepcMFNType*)
    int MFNSetOperator(SlepcMFN,PetscMat)
    int MFNGetOperator(SlepcMFN,PetscMat*)
    int MFNSetOptionsPrefix(SlepcMFN,char*)
    int MFNGetOptionsPrefix(SlepcMFN,char*[])
    int MFNSetFromOptions(SlepcMFN)
    int MFNAppendOptionsPrefix(SlepcMFN,char*)
    int MFNSetUp(SlepcMFN)
    int MFNSolve(SlepcMFN,PetscVec,PetscVec)
    int MFNSolveTranspose(SlepcMFN,PetscVec,PetscVec)

    int MFNSetBV(SlepcMFN,SlepcBV)
    int MFNGetBV(SlepcMFN,SlepcBV*)
    int MFNSetFN(SlepcMFN,SlepcFN)
    int MFNGetFN(SlepcMFN,SlepcFN*)
    int MFNSetTolerances(SlepcMFN,PetscReal,PetscInt)
    int MFNGetTolerances(SlepcMFN,PetscReal*,PetscInt*)
    int MFNSetDimensions(SlepcMFN,PetscInt)
    int MFNGetDimensions(SlepcMFN,PetscInt*)

    int MFNSetErrorIfNotConverged(SlepcMFN,PetscBool)
    int MFNGetErrorIfNotConverged(SlepcMFN,PetscBool*)

    int MFNMonitorSet(SlepcMFN,SlepcMFNMonitorFunction,void*,SlepcMFNCtxDel)
    int MFNMonitorCancel(SlepcMFN)
    int MFNGetIterationNumber(SlepcMFN,PetscInt*)

    int MFNGetConvergedReason(SlepcMFN,SlepcMFNConvergedReason*)

# -----------------------------------------------------------------------------

cdef inline MFN ref_MFN(SlepcMFN mfn):
    cdef MFN ob = <MFN> MFN()
    ob.mfn = mfn
    PetscINCREF(ob.obj)
    return ob

# -----------------------------------------------------------------------------

cdef int MFN_Monitor(
    SlepcMFN    mfn,
    PetscInt    it,
    PetscReal   errest,
    void        *ctx,
    ) except PETSC_ERR_PYTHON with gil:
    cdef MFN Mfn = ref_MFN(mfn)
    cdef object monitorlist = Mfn.get_attr('__monitor__')
    if monitorlist is None: return 0
    for (monitor, args, kargs) in monitorlist:
        monitor(Mfn, toInt(it), toReal(errest), *args, **kargs)
    return 0

# -----------------------------------------------------------------------------
