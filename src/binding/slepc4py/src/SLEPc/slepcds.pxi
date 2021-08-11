cdef extern from * nogil:

    ctypedef char* SlepcDSType "const char*"
    SlepcDSType DSHEP
    SlepcDSType DSNHEP
    SlepcDSType DSGHEP
    SlepcDSType DSGHIEP
    SlepcDSType DSGNHEP
    SlepcDSType DSNHEPTS
    SlepcDSType DSSVD
    SlepcDSType DSGSVD
    SlepcDSType DSPEP
    SlepcDSType DSNEP

    ctypedef enum SlepcDSStateType "DSStateType":
        DS_STATE_RAW
        DS_STATE_INTERMEDIATE
        DS_STATE_CONDENSED
        DS_STATE_TRUNCATED

    ctypedef enum SlepcDSMatType "DSMatType":
        DS_MAT_A
        DS_MAT_B
        DS_MAT_C
        DS_MAT_T
        DS_MAT_D
        DS_MAT_Q
        DS_MAT_Z
        DS_MAT_X
        DS_MAT_Y
        DS_MAT_U
        DS_MAT_V
        DS_MAT_W
        DS_NUM_MAT

    ctypedef enum SlepcDSParallelType "DSParallelType":
        DS_PARALLEL_REDUNDANT
        DS_PARALLEL_SYNCHRONIZED
        DS_PARALLEL_DISTRIBUTED

    int DSCreate(MPI_Comm,SlepcDS*)
    int DSView(SlepcDS,PetscViewer)
    int DSDestroy(SlepcDS*)
    int DSReset(SlepcDS)
    int DSSetType(SlepcDS,SlepcDSType)
    int DSGetType(SlepcDS,SlepcDSType*)

    int DSSetOptionsPrefix(SlepcDS,char[])
    int DSGetOptionsPrefix(SlepcDS,char*[])
    int DSAppendOptionsPrefix(SlepcDS,char[])
    int DSSetFromOptions(SlepcDS)
    int DSDuplicate(SlepcDS,SlepcDS*)

    int DSAllocate(SlepcDS,PetscInt)
    int DSGetLeadingDimension(SlepcDS,PetscInt*)
    int DSSetState(SlepcDS,SlepcDSStateType)
    int DSGetState(SlepcDS,SlepcDSStateType*)
    int DSSetDimensions(SlepcDS,PetscInt,PetscInt,PetscInt)
    int DSGetDimensions(SlepcDS,PetscInt*,PetscInt*,PetscInt*,PetscInt*)
    int DSTruncate(SlepcDS,PetscInt,PetscBool)
    int DSSetBlockSize(SlepcDS,PetscInt)
    int DSGetBlockSize(SlepcDS,PetscInt*)
    int DSSetMethod(SlepcDS,PetscInt)
    int DSGetMethod(SlepcDS,PetscInt*)
    int DSSetParallel(SlepcDS,SlepcDSParallelType)
    int DSGetParallel(SlepcDS,SlepcDSParallelType*)
    int DSSetCompact(SlepcDS,PetscBool)
    int DSGetCompact(SlepcDS,PetscBool*)
    int DSSetExtraRow(SlepcDS,PetscBool)
    int DSGetExtraRow(SlepcDS,PetscBool*)
    int DSSetRefined(SlepcDS,PetscBool)
    int DSGetRefined(SlepcDS,PetscBool*)
    int DSGetMat(SlepcDS,SlepcDSMatType,PetscMat*)
    int DSRestoreMat(SlepcDS,SlepcDSMatType,PetscMat*)
    int DSSetIdentity(SlepcDS,SlepcDSMatType)
    int DSVectors(SlepcDS,SlepcDSMatType,PetscInt*,PetscReal*)
    int DSSolve(SlepcDS,PetscScalar*,PetscScalar*)
    int DSSort(SlepcDS,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,PetscInt*)
    int DSUpdateExtraRow(SlepcDS)
    int DSCond(SlepcDS,PetscReal*)
    int DSTranslateHarmonic(SlepcDS,PetscScalar,PetscReal,PetscBool,PetscScalar*,PetscReal*)
    int DSTranslateRKS(SlepcDS,PetscScalar)
    int DSCond(SlepcDS,PetscReal*)
    int DSNormalize(SlepcDS,SlepcDSMatType,PetscInt)

    int DSSVDSetDimensions(SlepcDS,PetscInt)
    int DSSVDGetDimensions(SlepcDS,PetscInt*)
    int DSGSVDSetDimensions(SlepcDS,PetscInt,PetscInt)
    int DSGSVDGetDimensions(SlepcDS,PetscInt*,PetscInt*)

    int DSPEPSetDegree(SlepcDS,PetscInt)
    int DSPEPGetDegree(SlepcDS,PetscInt*)
    int DSPEPSetCoefficients(SlepcDS,PetscReal*)
    int DSPEPGetCoefficients(SlepcDS,PetscReal**)

