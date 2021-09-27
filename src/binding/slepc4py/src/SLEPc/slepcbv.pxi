cdef extern from * nogil:

    ctypedef char* SlepcBVType "const char*"
    SlepcBVType BVMAT
    SlepcBVType BVSVEC
    SlepcBVType BVVECS
    SlepcBVType BVCONTIGUOUS
    SlepcBVType BVTENSOR

    ctypedef enum SlepcBVOrthogType "BVOrthogType":
        BV_ORTHOG_CGS
        BV_ORTHOG_MGS

    ctypedef enum SlepcBVOrthogRefineType "BVOrthogRefineType":
        BV_ORTHOG_REFINE_IFNEEDED
        BV_ORTHOG_REFINE_NEVER
        BV_ORTHOG_REFINE_ALWAYS

    ctypedef enum SlepcBVOrthogBlockType "BVOrthogBlockType":
        BV_ORTHOG_BLOCK_GS
        BV_ORTHOG_BLOCK_CHOL
        BV_ORTHOG_BLOCK_TSQR
        BV_ORTHOG_BLOCK_TSQRCHOL
        BV_ORTHOG_BLOCK_SVQB

    ctypedef enum SlepcBVMatMultType "BVMatMultType":
        BV_MATMULT_VECS
        BV_MATMULT_MAT

    ctypedef enum SlepcBVSVDMethod "BVSVDMethod":
        BV_SVD_METHOD_REFINE
        BV_SVD_METHOD_QR
        BV_SVD_METHOD_QR_CAA

    int BVCreate(MPI_Comm,SlepcBV*)
    int BVCreateMat(SlepcBV,PetscMat*)
    int BVDuplicate(SlepcBV,SlepcBV*)
    int BVDuplicateResize(SlepcBV,PetscInt,SlepcBV*)
    int BVCopy(SlepcBV,SlepcBV)
    int BVView(SlepcBV,PetscViewer)
    int BVDestroy(SlepcBV*)
    int BVSetType(SlepcBV,SlepcBVType)
    int BVGetType(SlepcBV,SlepcBVType*)
    int BVSetSizes(SlepcBV,PetscInt,PetscInt,PetscInt)
    int BVSetSizesFromVec(SlepcBV,PetscVec,PetscInt)
    int BVGetSizes(SlepcBV,PetscInt*,PetscInt*,PetscInt*)
    int BVResize(SlepcBV,PetscInt,PetscBool)

    int BVSetOptionsPrefix(SlepcBV,char[])
    int BVGetOptionsPrefix(SlepcBV,char*[])
    int BVAppendOptionsPrefix(SlepcBV,char[])
    int BVSetFromOptions(SlepcBV)

    int BVSetOrthogonalization(SlepcBV,SlepcBVOrthogType,SlepcBVOrthogRefineType,PetscReal,SlepcBVOrthogBlockType)
    int BVGetOrthogonalization(SlepcBV,SlepcBVOrthogType*,SlepcBVOrthogRefineType*,PetscReal*,SlepcBVOrthogBlockType*)
    int BVSetMatMultMethod(SlepcBV,SlepcBVMatMultType)
    int BVGetMatMultMethod(SlepcBV,SlepcBVMatMultType*)

    int BVSetRandom(SlepcBV)
    int BVSetRandomNormal(SlepcBV)
    int BVSetRandomSign(SlepcBV)
    int BVSetRandomColumn(SlepcBV,PetscInt)
    int BVSetRandomCond(SlepcBV,PetscReal)
    int BVSetRandomContext(SlepcBV,PetscRandom)
    int BVGetRandomContext(SlepcBV,PetscRandom*)

    int BVSetMatrix(SlepcBV,PetscMat,PetscBool)
    int BVGetMatrix(SlepcBV,PetscMat*,PetscBool*)
    int BVApplyMatrix(SlepcBV,PetscVec,PetscVec)

    int BVSetActiveColumns(SlepcBV,PetscInt,PetscInt)
    int BVGetActiveColumns(SlepcBV,PetscInt*,PetscInt*)
    int BVSetDefiniteTolerance(SlepcBV,PetscReal)
    int BVGetDefiniteTolerance(SlepcBV,PetscReal*)

    int BVCreateVec(SlepcBV,PetscVec*)
    int BVInsertVec(SlepcBV,PetscInt,PetscVec)
    int BVInsertVecs(SlepcBV,PetscInt,PetscInt*,PetscVec*,PetscBool)
    int BVInsertConstraints(SlepcBV,PetscInt*,PetscVec*)
    int BVSetNumConstraints(SlepcBV,PetscInt)
    int BVGetNumConstraints(SlepcBV,PetscInt*)
    int BVGetColumn(SlepcBV,PetscInt,PetscVec*)
    int BVRestoreColumn(SlepcBV,PetscInt,PetscVec*)
    int BVCopyVec(SlepcBV,PetscInt,PetscVec)
    int BVCopyColumn(SlepcBV,PetscInt,PetscInt)

    int BVDot(SlepcBV,SlepcBV,PetscMat)
    int BVDotVec(SlepcBV,PetscVec,PetscScalar*)
    int BVDotColumn(SlepcBV,PetscInt,PetscScalar*)

    int BVMatProject(SlepcBV,PetscMat,SlepcBV,PetscMat)
    int BVMatMult(SlepcBV,PetscMat,SlepcBV)
    int BVMatMultHermitianTranspose(SlepcBV,PetscMat,SlepcBV)
    int BVMatMultColumn(SlepcBV,PetscMat,PetscInt)
    int BVMatMultTransposeColumn(SlepcBV,PetscMat,PetscInt)
    int BVMatMultHermitianTransposeColumn(SlepcBV,PetscMat,PetscInt)

    int BVMult(SlepcBV,PetscScalar,PetscScalar,SlepcBV,PetscMat)
    int BVMultColumn(SlepcBV,PetscScalar,PetscScalar,PetscInt,PetscScalar*)
    int BVMultInPlace(SlepcBV,PetscMat,PetscInt,PetscInt)
    int BVMultVec(SlepcBV,PetscScalar,PetscScalar,PetscVec,PetscScalar*)

    int BVScaleColumn(SlepcBV,PetscInt,PetscScalar)
    int BVScale(SlepcBV,PetscScalar)

    int BVNormColumn(SlepcBV,PetscInt,PetscNormType,PetscReal*)
    int BVNorm(SlepcBV,PetscNormType,PetscReal*)

    int BVOrthogonalizeVec(SlepcBV,PetscVec,PetscScalar*,PetscReal*,PetscBool*)
    int BVOrthogonalizeColumn(SlepcBV,PetscInt,PetscScalar*,PetscReal*,PetscBool*)
    int BVOrthonormalizeColumn(SlepcBV,PetscInt,PetscBool,PetscReal*,PetscBool*)
    int BVOrthogonalize(SlepcBV,PetscMat)

    int BVCreateFromMat(PetscMat,SlepcBV*)
    int BVGetMat(SlepcBV,PetscMat*)
    int BVRestoreMat(SlepcBV,PetscMat*)

cdef inline int BV_Sizes(
    object size,
    PetscInt *_n,
    PetscInt *_N,
    ) except -1:
    # unpack and get local and global sizes
    cdef PetscInt n=PETSC_DECIDE, N=PETSC_DECIDE
    cdef object on, oN
    try:
        on, oN = size
    except (TypeError, ValueError):
        on = None; oN = size
    if on is not None: n = asInt(on)
    if oN is not None: N = asInt(oN)
    if n==PETSC_DECIDE and N==PETSC_DECIDE: raise ValueError(
        "local and global sizes cannot be both 'DECIDE'")
    # return result to the caller
    if _n != NULL: _n[0] = n
    if _N != NULL: _N[0] = N
    return 0
