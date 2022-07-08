cdef extern from * nogil:

    ctypedef char* SlepcEPSType "const char*"
    SlepcEPSType EPSPOWER
    SlepcEPSType EPSSUBSPACE
    SlepcEPSType EPSARNOLDI
    SlepcEPSType EPSLANCZOS
    SlepcEPSType EPSKRYLOVSCHUR
    SlepcEPSType EPSGD
    SlepcEPSType EPSJD
    SlepcEPSType EPSRQCG
    SlepcEPSType EPSLOBPCG
    SlepcEPSType EPSCISS
    SlepcEPSType EPSLYAPII
    SlepcEPSType EPSLAPACK
    SlepcEPSType EPSARPACK
    SlepcEPSType EPSTRLAN
    SlepcEPSType EPSBLOPEX
    SlepcEPSType EPSPRIMME
    SlepcEPSType EPSFEAST
    SlepcEPSType EPSSCALAPACK
    SlepcEPSType EPSELPA
    SlepcEPSType EPSELEMENTAL
    SlepcEPSType EPSEVSL

    ctypedef enum SlepcEPSProblemType "EPSProblemType":
        EPS_HEP
        EPS_GHEP
        EPS_NHEP
        EPS_GNHEP
        EPS_PGNHEP
        EPS_GHIEP

    ctypedef enum SlepcEPSExtraction "EPSExtraction":
        EPS_RITZ
        EPS_HARMONIC
        EPS_HARMONIC_RELATIVE
        EPS_HARMONIC_RIGHT
        EPS_HARMONIC_LARGEST
        EPS_REFINED
        EPS_REFINED_HARMONIC

    ctypedef enum SlepcEPSWhich "EPSWhich":
        EPS_LARGEST_MAGNITUDE
        EPS_LARGEST_REAL
        EPS_LARGEST_IMAGINARY
        EPS_SMALLEST_MAGNITUDE
        EPS_SMALLEST_REAL
        EPS_SMALLEST_IMAGINARY
        EPS_TARGET_MAGNITUDE
        EPS_TARGET_REAL
        EPS_TARGET_IMAGINARY
        EPS_ALL
        EPS_WHICH_USER

    ctypedef enum SlepcEPSBalance "EPSBalance":
        EPS_BALANCE_NONE
        EPS_BALANCE_ONESIDE
        EPS_BALANCE_TWOSIDE
        EPS_BALANCE_USER

    ctypedef enum SlepcEPSErrorType "EPSErrorType":
        EPS_ERROR_ABSOLUTE
        EPS_ERROR_RELATIVE
        EPS_ERROR_BACKWARD

    ctypedef enum SlepcEPSConv "EPSConv":
        EPS_CONV_ABS
        EPS_CONV_REL
        EPS_CONV_NORM
        EPS_CONV_USER

    ctypedef enum SlepcEPSStop "EPSStop":
        EPS_STOP_BASIC
        EPS_STOP_USER

    ctypedef enum SlepcEPSConvergedReason "EPSConvergedReason":
        EPS_CONVERGED_TOL
        EPS_CONVERGED_USER
        EPS_DIVERGED_ITS
        EPS_DIVERGED_BREAKDOWN
        EPS_DIVERGED_SYMMETRY_LOST
        EPS_CONVERGED_ITERATING

    ctypedef int (*SlepcEPSCtxDel)(void*)
    ctypedef int (*SlepcEPSStoppingFunction)(SlepcEPS,
                                             PetscInt,
                                             PetscInt,
                                             PetscInt,
                                             PetscInt,
                                             SlepcEPSConvergedReason*,
                                             void*) except PETSC_ERR_PYTHON
    ctypedef int (*SlepcEPSMonitorFunction)(SlepcEPS,
                                            PetscInt,
                                            PetscInt,
                                            PetscScalar*,
                                            PetscScalar*,
                                            PetscReal*,
                                            PetscInt,
                                            void*) except PETSC_ERR_PYTHON
    ctypedef int (*SlepcEPSArbitraryFunction)(PetscScalar,
                                             PetscScalar,
                                             PetscVec,
                                             PetscVec,
                                             PetscScalar*,
                                             PetscScalar*,
                                             void*) except PETSC_ERR_PYTHON
    ctypedef int (*SlepcEPSComparisonFunction)(PetscScalar,
                                               PetscScalar,
                                               PetscScalar,
                                               PetscScalar,
                                               PetscInt*,
                                               void*) except PETSC_ERR_PYTHON

    int EPSView(SlepcEPS,PetscViewer)
    int EPSDestroy(SlepcEPS*)
    int EPSReset(SlepcEPS)
    int EPSCreate(MPI_Comm,SlepcEPS*)
    int EPSSetType(SlepcEPS,SlepcEPSType)
    int EPSGetType(SlepcEPS,SlepcEPSType*)
    int EPSSetOptionsPrefix(SlepcEPS,char[])
    int EPSAppendOptionsPrefix(SlepcEPS,char [])
    int EPSGetOptionsPrefix(SlepcEPS,char*[])
    int EPSSetFromOptions(SlepcEPS)

    int EPSSetProblemType(SlepcEPS,SlepcEPSProblemType)
    int EPSGetProblemType(SlepcEPS,SlepcEPSProblemType*)
    int EPSIsGeneralized(SlepcEPS,PetscBool*)
    int EPSIsHermitian(SlepcEPS,PetscBool*)
    int EPSIsPositive(SlepcEPS,PetscBool*)
    int EPSSetExtraction(SlepcEPS,SlepcEPSExtraction)
    int EPSGetExtraction(SlepcEPS,SlepcEPSExtraction*)
    int EPSSetBalance(SlepcEPS,SlepcEPSBalance,PetscInt,PetscReal)
    int EPSGetBalance(SlepcEPS,SlepcEPSBalance*,PetscInt*,PetscReal*)
    int EPSSetWhichEigenpairs(SlepcEPS,SlepcEPSWhich)
    int EPSGetWhichEigenpairs(SlepcEPS,SlepcEPSWhich*)
    int EPSSetTarget(SlepcEPS,PetscScalar)
    int EPSGetTarget(SlepcEPS,PetscScalar*)
    int EPSSetInterval(SlepcEPS,PetscReal,PetscReal)
    int EPSGetInterval(SlepcEPS,PetscReal*,PetscReal*)
    int EPSSetTolerances(SlepcEPS,PetscReal,PetscInt)
    int EPSGetTolerances(SlepcEPS,PetscReal*,PetscInt*)
    int EPSSetDimensions(SlepcEPS,PetscInt,PetscInt,PetscInt)
    int EPSGetDimensions(SlepcEPS,PetscInt*,PetscInt*,PetscInt*)

    int EPSSetBV(SlepcEPS,SlepcBV)
    int EPSGetBV(SlepcEPS,SlepcBV*)
    int EPSSetDS(SlepcEPS,SlepcDS)
    int EPSGetDS(SlepcEPS,SlepcDS*)
    int EPSSetST(SlepcEPS,SlepcST)
    int EPSGetST(SlepcEPS,SlepcST*)
    int EPSSetRG(SlepcEPS,SlepcRG)
    int EPSGetRG(SlepcEPS,SlepcRG*)

    int EPSSetOperators(SlepcEPS,PetscMat,PetscMat)
    int EPSGetOperators(SlepcEPS,PetscMat*,PetscMat*)

    int EPSSetTwoSided(SlepcEPS,PetscBool)
    int EPSGetTwoSided(SlepcEPS,PetscBool*)
    int EPSSetPurify(SlepcEPS,PetscBool)
    int EPSGetPurify(SlepcEPS,PetscBool*)

    int EPSSetConvergenceTest(SlepcEPS,SlepcEPSConv)
    int EPSGetConvergenceTest(SlepcEPS,SlepcEPSConv*)

    int EPSSetTrueResidual(SlepcEPS,PetscBool)
    int EPSGetTrueResidual(SlepcEPS,PetscBool*)

    int EPSSetTrackAll(SlepcEPS,PetscBool)
    int EPSGetTrackAll(SlepcEPS,PetscBool*)

    int EPSSetDeflationSpace(SlepcEPS,PetscInt,PetscVec*)
    int EPSSetInitialSpace(SlepcEPS,PetscInt,PetscVec*)
    int EPSSetLeftInitialSpace(SlepcEPS,PetscInt,PetscVec*)

    int EPSSetUp(SlepcEPS)
    int EPSSolve(SlepcEPS)

    int EPSGetIterationNumber(SlepcEPS,PetscInt*)
    int EPSGetConvergedReason(SlepcEPS,SlepcEPSConvergedReason*)
    int EPSGetConverged(SlepcEPS,PetscInt*)
    int EPSGetEigenvalue(SlepcEPS,PetscInt,PetscScalar*,PetscScalar*)
    int EPSGetEigenvector(SlepcEPS,PetscInt,PetscVec,PetscVec)
    int EPSGetLeftEigenvector(SlepcEPS,PetscInt,PetscVec,PetscVec)
    int EPSGetEigenpair(SlepcEPS,PetscInt,PetscScalar*,PetscScalar*,PetscVec,PetscVec)
    int EPSGetInvariantSubspace(SlepcEPS,PetscVec*)

    int EPSSetStoppingTestFunction(SlepcEPS,SlepcEPSStoppingFunction,void*,SlepcEPSCtxDel)
    int EPSStoppingBasic(SlepcEPS,PetscInt,PetscInt,PetscInt,PetscInt,SlepcEPSConvergedReason*,void*) except PETSC_ERR_PYTHON

    int EPSSetArbitrarySelection(SlepcEPS,SlepcEPSArbitraryFunction,void*);
    int EPSSetEigenvalueComparison(SlepcEPS,SlepcEPSComparisonFunction,void*);

    int EPSGetErrorEstimate(SlepcEPS,PetscInt,PetscReal*)
    int EPSComputeError(SlepcEPS,PetscInt,SlepcEPSErrorType,PetscReal*)
    int EPSErrorView(SlepcEPS,SlepcEPSErrorType,PetscViewer)
    int EPSValuesView(SlepcEPS,PetscViewer)
    int EPSVectorsView(SlepcEPS,PetscViewer)

    int EPSMonitorSet(SlepcEPS,SlepcEPSMonitorFunction,void*,SlepcEPSCtxDel)
    int EPSMonitorCancel(SlepcEPS)

    ctypedef enum SlepcEPSPowerShiftType "EPSPowerShiftType":
        EPS_POWER_SHIFT_CONSTANT
        EPS_POWER_SHIFT_RAYLEIGH
        EPS_POWER_SHIFT_WILKINSON
    int EPSPowerSetShiftType(SlepcEPS,SlepcEPSPowerShiftType)
    int EPSPowerGetShiftType(SlepcEPS,SlepcEPSPowerShiftType*)

    int EPSArnoldiSetDelayed(SlepcEPS,PetscBool)
    int EPSArnoldiGetDelayed(SlepcEPS,PetscBool*)

    int EPSKrylovSchurSetRestart(SlepcEPS,PetscReal)
    int EPSKrylovSchurGetRestart(SlepcEPS,PetscReal*)
    int EPSKrylovSchurSetLocking(SlepcEPS,PetscBool)
    int EPSKrylovSchurGetLocking(SlepcEPS,PetscBool*)
    int EPSKrylovSchurSetPartitions(SlepcEPS,PetscInt)
    int EPSKrylovSchurGetPartitions(SlepcEPS,PetscInt*)
    int EPSKrylovSchurSetDetectZeros(SlepcEPS,PetscBool)
    int EPSKrylovSchurGetDetectZeros(SlepcEPS,PetscBool*)
    int EPSKrylovSchurSetDimensions(SlepcEPS,PetscInt,PetscInt,PetscInt)
    int EPSKrylovSchurGetDimensions(SlepcEPS,PetscInt*,PetscInt*,PetscInt*)
    int EPSKrylovSchurGetSubcommInfo(SlepcEPS,PetscInt*,PetscInt*,PetscVec*)
    int EPSKrylovSchurGetSubcommPairs(SlepcEPS,PetscInt,PetscScalar*,PetscVec)
    int EPSKrylovSchurGetSubcommMats(SlepcEPS,PetscMat*,PetscMat*)
    int EPSKrylovSchurUpdateSubcommMats(SlepcEPS,PetscScalar,PetscScalar,PetscMat,PetscScalar,PetscScalar,PetscMat,PetscMatStructure,PetscBool)
    int EPSKrylovSchurSetSubintervals(SlepcEPS,PetscReal*)
    int EPSKrylovSchurGetSubintervals(SlepcEPS,PetscReal**)
    int EPSKrylovSchurGetInertias(SlepcEPS,PetscInt*,PetscReal**,PetscInt**)
    int EPSKrylovSchurGetKSP(SlepcEPS,PetscKSP*)

    ctypedef enum SlepcEPSLanczosReorthogType "EPSLanczosReorthogType":
        EPS_LANCZOS_REORTHOG_LOCAL
        EPS_LANCZOS_REORTHOG_FULL
        EPS_LANCZOS_REORTHOG_SELECTIVE
        EPS_LANCZOS_REORTHOG_PERIODIC
        EPS_LANCZOS_REORTHOG_PARTIAL
        EPS_LANCZOS_REORTHOG_DELAYED
    int EPSLanczosSetReorthog(SlepcEPS,SlepcEPSLanczosReorthogType)
    int EPSLanczosGetReorthog(SlepcEPS,SlepcEPSLanczosReorthogType*)

    int EPSGDSetKrylovStart(SlepcEPS,PetscBool)
    int EPSGDGetKrylovStart(SlepcEPS,PetscBool*)
    int EPSGDSetBlockSize(SlepcEPS,PetscInt)
    int EPSGDGetBlockSize(SlepcEPS,PetscInt*)
    int EPSGDSetRestart(SlepcEPS,PetscInt,PetscInt)
    int EPSGDGetRestart(SlepcEPS,PetscInt*,PetscInt*)
    int EPSGDSetInitialSize(SlepcEPS,PetscInt)
    int EPSGDGetInitialSize(SlepcEPS,PetscInt*)
    int EPSGDSetBOrth(SlepcEPS,PetscBool)
    int EPSGDGetBOrth(SlepcEPS,PetscBool*)
    int EPSGDSetDoubleExpansion(SlepcEPS,PetscBool)
    int EPSGDGetDoubleExpansion(SlepcEPS,PetscBool*)

    int EPSJDSetKrylovStart(SlepcEPS,PetscBool)
    int EPSJDGetKrylovStart(SlepcEPS,PetscBool*)
    int EPSJDSetBlockSize(SlepcEPS,PetscInt)
    int EPSJDGetBlockSize(SlepcEPS,PetscInt*)
    int EPSJDSetRestart(SlepcEPS,PetscInt,PetscInt)
    int EPSJDGetRestart(SlepcEPS,PetscInt*,PetscInt*)
    int EPSJDSetInitialSize(SlepcEPS,PetscInt)
    int EPSJDGetInitialSize(SlepcEPS,PetscInt*)
    int EPSJDSetFix(SlepcEPS,PetscReal)
    int EPSJDGetFix(SlepcEPS,PetscReal*)
    int EPSJDSetConstCorrectionTol(SlepcEPS,PetscBool)
    int EPSJDGetConstCorrectionTol(SlepcEPS,PetscBool*)
    int EPSJDSetBOrth(SlepcEPS,PetscBool)
    int EPSJDGetBOrth(SlepcEPS,PetscBool*)

    int EPSRQCGSetReset(SlepcEPS,PetscInt)
    int EPSRQCGGetReset(SlepcEPS,PetscInt*)

    int EPSLOBPCGSetBlockSize(SlepcEPS,PetscInt)
    int EPSLOBPCGGetBlockSize(SlepcEPS,PetscInt*)
    int EPSLOBPCGSetRestart(SlepcEPS,PetscReal)
    int EPSLOBPCGGetRestart(SlepcEPS,PetscReal*)
    int EPSLOBPCGSetLocking(SlepcEPS,PetscBool)
    int EPSLOBPCGGetLocking(SlepcEPS,PetscBool*)

    int EPSLyapIISetRanks(SlepcEPS,PetscInt,PetscInt)
    int EPSLyapIIGetRanks(SlepcEPS,PetscInt*,PetscInt*)

    ctypedef enum SlepcEPSCISSQuadRule "EPSCISSQuadRule":
        EPS_CISS_QUADRULE_TRAPEZOIDAL
        EPS_CISS_QUADRULE_CHEBYSHEV

    ctypedef enum SlepcEPSCISSExtraction "EPSCISSExtraction":
        EPS_CISS_EXTRACTION_RITZ
        EPS_CISS_EXTRACTION_HANKEL

    int EPSCISSSetExtraction(SlepcEPS,SlepcEPSCISSExtraction)
    int EPSCISSGetExtraction(SlepcEPS,SlepcEPSCISSExtraction*)
    int EPSCISSSetQuadRule(SlepcEPS,SlepcEPSCISSQuadRule)
    int EPSCISSGetQuadRule(SlepcEPS,SlepcEPSCISSQuadRule*)
    int EPSCISSSetSizes(SlepcEPS,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,PetscBool)
    int EPSCISSGetSizes(SlepcEPS,PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscBool*)
    int EPSCISSSetThreshold(SlepcEPS,PetscReal,PetscReal)
    int EPSCISSGetThreshold(SlepcEPS,PetscReal*,PetscReal*)
    int EPSCISSSetRefinement(SlepcEPS,PetscInt,PetscInt)
    int EPSCISSGetRefinement(SlepcEPS,PetscInt*,PetscInt*)
    int EPSCISSSetUseST(SlepcEPS,PetscBool)
    int EPSCISSGetUseST(SlepcEPS,PetscBool*)
    int EPSCISSGetKSPs(SlepcEPS,PetscInt*,PetscKSP**)

cdef extern from * nogil:
    int VecDuplicate(PetscVec,PetscVec*)
    int MatCreateVecs(PetscMat,PetscVec*,PetscVec*)

# -----------------------------------------------------------------------------

cdef inline EPS ref_EPS(SlepcEPS eps):
    cdef EPS ob = <EPS> EPS()
    ob.eps = eps
    PetscINCREF(ob.obj)
    return ob

# -----------------------------------------------------------------------------

cdef int EPS_Stopping(
    SlepcEPS                eps,
    PetscInt                its,
    PetscInt                max_it,
    PetscInt                nconv,
    PetscInt                nev,
    SlepcEPSConvergedReason *r,
    void                    *ctx,
    ) except PETSC_ERR_PYTHON with gil:
    cdef EPS Eps = ref_EPS(eps)
    (stopping, args, kargs) = Eps.get_attr('__stopping__')
    reason = stopping(Eps, toInt(its), toInt(max_it), toInt(nconv), toInt(nev), *args, **kargs)
    if   reason is None:  r[0] = EPS_CONVERGED_ITERATING
    elif reason is False: r[0] = EPS_CONVERGED_ITERATING
    elif reason is True:  r[0] = EPS_CONVERGED_USER
    else:                 r[0] = reason

# -----------------------------------------------------------------------------

cdef int EPS_Arbitrary(
    PetscScalar  er,
    PetscScalar  ei,
    PetscVec     xr,
    PetscVec     xi,
    PetscScalar* rr,
    PetscScalar* ri,
    void         *ctx,
    ) except PETSC_ERR_PYTHON with gil:
    (arbitrary, args, kargs) = <object>ctx
    cdef Vec Vr = ref_Vec(xr)
    cdef Vec Vi = ref_Vec(xi)
    r = arbitrary(toComplex(er,ei), Vr, Vi, args, **kargs)
    if sizeof(PetscScalar) == sizeof(PetscReal):
        rr[0] = asComplexReal(r)
        ri[0] = asComplexImag(r)
    else:
        rr[0] = asScalar(r)
        ri[0] = 0.0

# -----------------------------------------------------------------------------

cdef int EPS_Comparison(
    PetscScalar  ar,
    PetscScalar  ai,
    PetscScalar  br,
    PetscScalar  bi,
    PetscInt*    res,
    void         *ctx,
    ) except PETSC_ERR_PYTHON with gil:
    (comparison, args, kargs) = <object>ctx
    r = comparison(toComplex(ar, ai), toComplex(br, bi), args, **kargs)
    res[0] = asInt(r)

# -----------------------------------------------------------------------------

cdef int EPS_Monitor(
    SlepcEPS    eps,
    PetscInt    its,
    PetscInt    nconv,
    PetscScalar *eigr,
    PetscScalar *eigi,
    PetscReal   *errest,
    PetscInt    nest,
    void        *ctx,
    ) except PETSC_ERR_PYTHON with gil:
    cdef EPS Eps = ref_EPS(eps)
    cdef object monitorlist = Eps.get_attr('__monitor__')
    if monitorlist is None: return 0
    cdef object eig = [toComplex(eigr[i], eigi[i]) for i in range(nest)]
    cdef object err = [toReal(errest[i]) for i in range(nest)]
    for (monitor, args, kargs) in monitorlist:
        monitor(Eps, toInt(its), toInt(nconv), eig, err, *args, **kargs)
    return 0

# -----------------------------------------------------------------------------
