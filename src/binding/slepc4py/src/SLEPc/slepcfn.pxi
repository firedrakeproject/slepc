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

    int FNCreate(MPI_Comm,SlepcFN*)
    int FNView(SlepcFN,PetscViewer)
    int FNDestroy(SlepcFN*)
    int FNReset(SlepcFN)
    int FNSetType(SlepcFN,SlepcFNType)
    int FNGetType(SlepcFN,SlepcFNType*)

    int FNSetOptionsPrefix(SlepcFN,char[])
    int FNGetOptionsPrefix(SlepcFN,char*[])
    int FNAppendOptionsPrefix(SlepcFN,char[])
    int FNSetFromOptions(SlepcFN)
    int FNDuplicate(SlepcFN,MPI_Comm,SlepcFN*)

    int FNSetScale(SlepcFN,PetscScalar,PetscScalar)
    int FNGetScale(SlepcFN,PetscScalar*,PetscScalar*)
    int FNSetMethod(SlepcFN,PetscInt)
    int FNGetMethod(SlepcFN,PetscInt*)
    int FNSetParallel(SlepcFN,SlepcFNParallelType)
    int FNGetParallel(SlepcFN,SlepcFNParallelType*)
    int FNEvaluateFunction(SlepcFN,PetscScalar,PetscScalar*)
    int FNEvaluateDerivative(SlepcFN,PetscScalar,PetscScalar*)
    int FNEvaluateFunctionMat(SlepcFN,PetscMat,PetscMat)
    int FNEvaluateFunctionMatVec(SlepcFN,PetscMat,PetscVec)

    int FNRationalSetNumerator(SlepcFN,PetscInt,PetscScalar[])
    int FNRationalGetNumerator(SlepcFN,PetscInt*,PetscScalar*[])
    int FNRationalSetDenominator(SlepcFN,PetscInt,PetscScalar[])
    int FNRationalGetDenominator(SlepcFN,PetscInt*,PetscScalar*[])

    int FNCombineSetChildren(SlepcFN,SlepcFNCombineType,SlepcFN,SlepcFN)
    int FNCombineGetChildren(SlepcFN,SlepcFNCombineType*,SlepcFN*,SlepcFN*)

    int FNPhiSetIndex(SlepcFN,PetscInt)
    int FNPhiGetIndex(SlepcFN,PetscInt*)

