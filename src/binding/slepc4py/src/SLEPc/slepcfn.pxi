cdef extern from * nogil:

    ctypedef char* SlepcFNType "const char*"
    SlepcFNType FNCOMBINE
    SlepcFNType FNRATIONAL
    SlepcFNType FNEXP
    SlepcFNType FNLOG
    SlepcFNType FNPHI
    SlepcFNType FNSQRT
    SlepcFNType FNINVSQRT

    ctypedef enum SlepcFNCombineType "FNCombineType":
        FN_COMBINE_ADD
        FN_COMBINE_MULTIPLY
        FN_COMBINE_DIVIDE
        FN_COMBINE_COMPOSE

    ctypedef enum SlepcFNParallelType "FNParallelType":
        FN_PARALLEL_REDUNDANT
        FN_PARALLEL_SYNCHRONIZED

    PetscErrorCode FNCreate(MPI_Comm,SlepcFN*)
    PetscErrorCode FNView(SlepcFN,PetscViewer)
    PetscErrorCode FNDestroy(SlepcFN*)
    PetscErrorCode FNReset(SlepcFN)
    PetscErrorCode FNSetType(SlepcFN,SlepcFNType)
    PetscErrorCode FNGetType(SlepcFN,SlepcFNType*)

    PetscErrorCode FNSetOptionsPrefix(SlepcFN,char[])
    PetscErrorCode FNGetOptionsPrefix(SlepcFN,char*[])
    PetscErrorCode FNAppendOptionsPrefix(SlepcFN,char[])
    PetscErrorCode FNSetFromOptions(SlepcFN)
    PetscErrorCode FNDuplicate(SlepcFN,MPI_Comm,SlepcFN*)

    PetscErrorCode FNSetScale(SlepcFN,PetscScalar,PetscScalar)
    PetscErrorCode FNGetScale(SlepcFN,PetscScalar*,PetscScalar*)
    PetscErrorCode FNSetMethod(SlepcFN,PetscInt)
    PetscErrorCode FNGetMethod(SlepcFN,PetscInt*)
    PetscErrorCode FNSetParallel(SlepcFN,SlepcFNParallelType)
    PetscErrorCode FNGetParallel(SlepcFN,SlepcFNParallelType*)
    PetscErrorCode FNEvaluateFunction(SlepcFN,PetscScalar,PetscScalar*)
    PetscErrorCode FNEvaluateDerivative(SlepcFN,PetscScalar,PetscScalar*)
    PetscErrorCode FNEvaluateFunctionMat(SlepcFN,PetscMat,PetscMat)
    PetscErrorCode FNEvaluateFunctionMatVec(SlepcFN,PetscMat,PetscVec)

    PetscErrorCode FNRationalSetNumerator(SlepcFN,PetscInt,PetscScalar[])
    PetscErrorCode FNRationalGetNumerator(SlepcFN,PetscInt*,PetscScalar*[])
    PetscErrorCode FNRationalSetDenominator(SlepcFN,PetscInt,PetscScalar[])
    PetscErrorCode FNRationalGetDenominator(SlepcFN,PetscInt*,PetscScalar*[])

    PetscErrorCode FNCombineSetChildren(SlepcFN,SlepcFNCombineType,SlepcFN,SlepcFN)
    PetscErrorCode FNCombineGetChildren(SlepcFN,SlepcFNCombineType*,SlepcFN*,SlepcFN*)

    PetscErrorCode FNPhiSetIndex(SlepcFN,PetscInt)
    PetscErrorCode FNPhiGetIndex(SlepcFN,PetscInt*)

