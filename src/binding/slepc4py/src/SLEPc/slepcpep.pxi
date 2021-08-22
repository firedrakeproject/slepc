cdef extern from * nogil:

    ctypedef char* SlepcPEPType "const char*"
    SlepcPEPType PEPLINEAR
    SlepcPEPType PEPQARNOLDI
    SlepcPEPType PEPTOAR
    SlepcPEPType PEPSTOAR
    SlepcPEPType PEPJD
    SlepcPEPType PEPCISS

    ctypedef enum SlepcPEPProblemType "PEPProblemType":
        PEP_GENERAL
        PEP_HERMITIAN
        PEP_HYPERBOLIC
        PEP_GYROSCOPIC

    ctypedef enum SlepcPEPRefine "PEPRefine":
        PEP_REFINE_NONE
        PEP_REFINE_SIMPLE
        PEP_REFINE_MULTIPLE

    ctypedef enum SlepcPEPExtract "PEPExtract":
        PEP_EXTRACT_NONE
        PEP_EXTRACT_NORM
        PEP_EXTRACT_RESIDUAL
        PEP_EXTRACT_STRUCTURED

    ctypedef enum SlepcPEPRefineScheme "PEPRefineScheme":
        PEP_REFINE_SCHEME_EXPLICIT
        PEP_REFINE_SCHEME_MBE
        PEP_REFINE_SCHEME_SCHUR

    ctypedef enum SlepcPEPErrorType "PEPErrorType":
        PEP_ERROR_ABSOLUTE
        PEP_ERROR_RELATIVE
        PEP_ERROR_BACKWARD

    ctypedef enum SlepcPEPWhich "PEPWhich":
        PEP_LARGEST_MAGNITUDE
        PEP_SMALLEST_MAGNITUDE
        PEP_LARGEST_REAL
        PEP_SMALLEST_REAL
        PEP_LARGEST_IMAGINARY
        PEP_SMALLEST_IMAGINARY
        PEP_TARGET_MAGNITUDE
        PEP_TARGET_REAL
        PEP_TARGET_IMAGINARY
        PEP_ALL
        PEP_WHICH_USER

    ctypedef enum SlepcPEPBasis "PEPBasis":
        PEP_BASIS_MONOMIAL
        PEP_BASIS_CHEBYSHEV1
        PEP_BASIS_CHEBYSHEV2
        PEP_BASIS_LEGENDRE
        PEP_BASIS_LAGUERRE
        PEP_BASIS_HERMITE

    ctypedef enum SlepcPEPScale "PEPScale":
        PEP_SCALE_NONE
        PEP_SCALE_SCALAR
        PEP_SCALE_DIAGONAL
        PEP_SCALE_BOTH

    ctypedef enum SlepcPEPConv "PEPConv":
        PEP_CONV_ABS
        PEP_CONV_REL
        PEP_CONV_NORM
        PEP_CONV_USER

    ctypedef enum SlepcPEPStop "PEPStop":
        PEP_STOP_BASIC
        PEP_STOP_USER

    ctypedef enum SlepcPEPConvergedReason "PEPConvergedReason":
        PEP_CONVERGED_TOL
        PEP_CONVERGED_USER
        PEP_DIVERGED_ITS
        PEP_DIVERGED_BREAKDOWN
        PEP_DIVERGED_SYMMETRY_LOST
        PEP_CONVERGED_ITERATING

    ctypedef int (*SlepcPEPCtxDel)(void*)
    ctypedef int (*SlepcPEPStoppingFunction)(SlepcPEP,
                                             PetscInt,
                                             PetscInt,
                                             PetscInt,
                                             PetscInt,
                                             SlepcPEPConvergedReason*,
                                             void*) except PETSC_ERR_PYTHON
    ctypedef int (*SlepcPEPMonitorFunction)(SlepcPEP,
                                            PetscInt,
                                            PetscInt,
                                            PetscScalar*,
                                            PetscScalar*,
                                            PetscReal*,
                                            PetscInt,
                                            void*) except PETSC_ERR_PYTHON

    int PEPCreate(MPI_Comm,SlepcPEP*)
    int PEPDestroy(SlepcPEP*)
    int PEPReset(SlepcPEP)
    int PEPView(SlepcPEP,PetscViewer)

    int PEPSetType(SlepcPEP,SlepcPEPType)
    int PEPGetType(SlepcPEP,SlepcPEPType*)
    int PEPSetBasis(SlepcPEP,SlepcPEPBasis)
    int PEPGetBasis(SlepcPEP,SlepcPEPBasis*)
    int PEPSetProblemType(SlepcPEP,SlepcPEPProblemType)
    int PEPGetProblemType(SlepcPEP,SlepcPEPProblemType*)
    int PEPSetOperators(SlepcPEP,PetscInt,PetscMat*)
    int PEPGetOperators(SlepcPEP,PetscInt,PetscMat*)
    int PEPGetNumMatrices(SlepcPEP,PetscInt*)
    int PEPSetOptionsPrefix(SlepcPEP,char*)
    int PEPGetOptionsPrefix(SlepcPEP,char*[])
    int PEPSetFromOptions(SlepcPEP)
    int PEPAppendOptionsPrefix(SlepcPEP,char*)
    int PEPSetUp(SlepcPEP)
    int PEPSolve(SlepcPEP)

    int PEPSetBV(SlepcPEP,SlepcBV)
    int PEPGetBV(SlepcPEP,SlepcBV*)
    int PEPSetDS(SlepcPEP,SlepcDS)
    int PEPGetDS(SlepcPEP,SlepcDS*)
    int PEPSetST(SlepcPEP,SlepcST)
    int PEPGetST(SlepcPEP,SlepcST*)
    int PEPSetRG(SlepcPEP,SlepcRG)
    int PEPGetRG(SlepcPEP,SlepcRG*)

    int PEPSetTrackAll(SlepcPEP,PetscBool)
    int PEPGetTrackAll(SlepcPEP,PetscBool*)

    int PEPSetTolerances(SlepcPEP,PetscReal,PetscInt)
    int PEPGetTolerances(SlepcPEP,PetscReal*,PetscInt*)
    int PEPSetDimensions(SlepcPEP,PetscInt,PetscInt,PetscInt)
    int PEPGetDimensions(SlepcPEP,PetscInt*,PetscInt*,PetscInt*)
    int PEPSetScale(SlepcPEP,SlepcPEPScale,PetscReal,PetscVec,PetscVec,PetscInt,PetscReal)
    int PEPGetScale(SlepcPEP,SlepcPEPScale*,PetscReal*,PetscVec*,PetscVec*,PetscInt*,PetscReal*)

    int PEPGetConverged(SlepcPEP,PetscInt*)
    int PEPGetEigenpair(SlepcPEP,PetscInt,PetscScalar*,PetscScalar*,PetscVec,PetscVec)
    int PEPComputeError(SlepcPEP,PetscInt,SlepcPEPErrorType,PetscReal*)
    int PEPErrorView(SlepcPEP,SlepcPEPErrorType,PetscViewer)
    int PEPValuesView(SlepcPEP,PetscViewer)
    int PEPVectorsView(SlepcPEP,PetscViewer)
    int PEPGetErrorEstimate(SlepcPEP,PetscInt,PetscReal*)

    int PEPSetStoppingTestFunction(SlepcPEP,SlepcPEPStoppingFunction,void*,SlepcPEPCtxDel)
    int PEPStoppingBasic(SlepcPEP,PetscInt,PetscInt,PetscInt,PetscInt,SlepcPEPConvergedReason*,void*) except PETSC_ERR_PYTHON

    int PEPSetConvergenceTest(SlepcPEP,SlepcPEPConv)
    int PEPGetConvergenceTest(SlepcPEP,SlepcPEPConv*)
    int PEPSetRefine(SlepcPEP,SlepcPEPRefine,PetscInt,PetscReal,PetscInt,SlepcPEPRefineScheme)
    int PEPGetRefine(SlepcPEP,SlepcPEPRefine*,PetscInt*,PetscReal*,PetscInt*,SlepcPEPRefineScheme*)
    int PEPRefineGetKSP(SlepcPEP,PetscKSP*)
    int PEPSetExtract(SlepcPEP,SlepcPEPExtract);
    int PEPGetExtract(SlepcPEP,SlepcPEPExtract*)

    int PEPMonitorSet(SlepcPEP,SlepcPEPMonitorFunction,void*,SlepcPEPCtxDel)
    int PEPMonitorCancel(SlepcPEP)
    int PEPGetIterationNumber(SlepcPEP,PetscInt*)

    int PEPSetInitialSpace(SlepcPEP,PetscInt,PetscVec*)
    int PEPSetWhichEigenpairs(SlepcPEP,SlepcPEPWhich)
    int PEPGetWhichEigenpairs(SlepcPEP,SlepcPEPWhich*)
    int PEPSetTarget(SlepcPEP,PetscScalar)
    int PEPGetTarget(SlepcPEP,PetscScalar*)
    int PEPSetInterval(SlepcPEP,PetscReal,PetscReal)
    int PEPGetInterval(SlepcPEP,PetscReal*,PetscReal*)
    int PEPGetConvergedReason(SlepcPEP,SlepcPEPConvergedReason*)

    int PEPLinearSetLinearization(SlepcPEP,PetscReal,PetscReal)
    int PEPLinearGetLinearization(SlepcPEP,PetscReal*,PetscReal*)
    int PEPLinearSetExplicitMatrix(SlepcPEP,PetscBool)
    int PEPLinearGetExplicitMatrix(SlepcPEP,PetscBool*)
    int PEPLinearSetEPS(SlepcPEP,SlepcEPS)
    int PEPLinearGetEPS(SlepcPEP,SlepcEPS*)

    int PEPQArnoldiSetRestart(SlepcPEP,PetscReal)
    int PEPQArnoldiGetRestart(SlepcPEP,PetscReal*)
    int PEPQArnoldiSetLocking(SlepcPEP,PetscBool)
    int PEPQArnoldiGetLocking(SlepcPEP,PetscBool*)

    int PEPTOARSetRestart(SlepcPEP,PetscReal)
    int PEPTOARGetRestart(SlepcPEP,PetscReal*)
    int PEPTOARSetLocking(SlepcPEP,PetscBool)
    int PEPTOARGetLocking(SlepcPEP,PetscBool*)

    int PEPSTOARSetLinearization(SlepcPEP,PetscReal,PetscReal)
    int PEPSTOARGetLinearization(SlepcPEP,PetscReal*,PetscReal*)
    int PEPSTOARSetLocking(SlepcPEP,PetscBool)
    int PEPSTOARGetLocking(SlepcPEP,PetscBool*)
    int PEPSTOARSetDetectZeros(SlepcPEP,PetscBool)
    int PEPSTOARGetDetectZeros(SlepcPEP,PetscBool*)
    int PEPSTOARSetDimensions(SlepcPEP,PetscInt,PetscInt,PetscInt)
    int PEPSTOARGetDimensions(SlepcPEP,PetscInt*,PetscInt*,PetscInt*)
    int PEPSTOARGetInertias(SlepcPEP,PetscInt*,PetscReal**,PetscInt**)
    int PEPSTOARSetCheckEigenvalueType(SlepcPEP,PetscBool)
    int PEPSTOARGetCheckEigenvalueType(SlepcPEP,PetscBool*)

    ctypedef enum SlepcPEPJDProjection "PEPJDProjection":
        PEP_JD_PROJECTION_HARMONIC
        PEP_JD_PROJECTION_ORTHOGONAL

    int PEPJDSetRestart(SlepcPEP,PetscReal)
    int PEPJDGetRestart(SlepcPEP,PetscReal*)
    int PEPJDSetFix(SlepcPEP,PetscReal)
    int PEPJDGetFix(SlepcPEP,PetscReal*)
    int PEPJDSetReusePreconditioner(SlepcPEP,PetscBool)
    int PEPJDGetReusePreconditioner(SlepcPEP,PetscBool*)
    int PEPJDSetMinimalityIndex(SlepcPEP,PetscInt)
    int PEPJDGetMinimalityIndex(SlepcPEP,PetscInt*)
    int PEPJDSetProjection(SlepcPEP,SlepcPEPJDProjection)
    int PEPJDGetProjection(SlepcPEP,SlepcPEPJDProjection*)

    ctypedef enum SlepcPEPCISSExtraction "PEPCISSExtraction":
        PEP_CISS_EXTRACTION_RITZ
        PEP_CISS_EXTRACTION_HANKEL
        PEP_CISS_EXTRACTION_CAA

    int PEPCISSSetExtraction(SlepcPEP,SlepcPEPCISSExtraction)
    int PEPCISSGetExtraction(SlepcPEP,SlepcPEPCISSExtraction*)
    int PEPCISSSetSizes(SlepcPEP,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,PetscBool)
    int PEPCISSGetSizes(SlepcPEP,PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscBool*)
    int PEPCISSSetThreshold(SlepcPEP,PetscReal,PetscReal)
    int PEPCISSGetThreshold(SlepcPEP,PetscReal*,PetscReal*)
    int PEPCISSSetRefinement(SlepcPEP,PetscInt,PetscInt)
    int PEPCISSGetRefinement(SlepcPEP,PetscInt*,PetscInt*)
    int PEPCISSGetKSPs(SlepcPEP,PetscInt*,PetscKSP**)

# -----------------------------------------------------------------------------

cdef inline PEP ref_PEP(SlepcPEP pep):
    cdef PEP ob = <PEP> PEP()
    ob.pep = pep
    PetscINCREF(ob.obj)
    return ob

# -----------------------------------------------------------------------------

cdef int PEP_Stopping(
    SlepcPEP                pep,
    PetscInt                its,
    PetscInt                max_it,
    PetscInt                nconv,
    PetscInt                nev,
    SlepcPEPConvergedReason *r,
    void                    *ctx,
    ) except PETSC_ERR_PYTHON with gil:
    cdef PEP Pep = ref_PEP(pep)
    (stopping, args, kargs) = Pep.get_attr('__stopping__')
    reason = stopping(Pep, toInt(its), toInt(max_it), toInt(nconv), toInt(nev), *args, **kargs)
    if   reason is None:  r[0] = PEP_CONVERGED_ITERATING
    elif reason is False: r[0] = PEP_CONVERGED_ITERATING
    elif reason is True:  r[0] = PEP_CONVERGED_USER
    else:                 r[0] = reason

# -----------------------------------------------------------------------------

cdef int PEP_Monitor(
    SlepcPEP    pep,
    PetscInt    its,
    PetscInt    nconv,
    PetscScalar *eigr,
    PetscScalar *eigi,
    PetscReal   *errest,
    PetscInt    nest,
    void        *ctx,
    ) except PETSC_ERR_PYTHON with gil:
    cdef PEP Pep = ref_PEP(pep)
    cdef object monitorlist = Pep.get_attr('__monitor__')
    if monitorlist is None: return 0
    cdef object eig = [toComplex(eigr[i], eigi[i]) for i in range(nest)]
    cdef object err = [toReal(errest[i]) for i in range(nest)]
    for (monitor, args, kargs) in monitorlist:
        monitor(Pep, toInt(its), toInt(nconv), eig, err, *args, **kargs)
    return 0

# -----------------------------------------------------------------------------
