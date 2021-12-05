cdef extern from * nogil:

    ctypedef char* SlepcNEPType "const char*"
    SlepcNEPType NEPRII
    SlepcNEPType NEPSLP
    SlepcNEPType NEPNARNOLDI
    SlepcNEPType NEPCISS
    SlepcNEPType NEPINTERPOL
    SlepcNEPType NEPNLEIGS

    ctypedef enum SlepcNEPProblemType "NEPProblemType":
        NEP_GENERAL
        NEP_RATIONAL

    ctypedef enum SlepcNEPWhich "NEPWhich":
        NEP_LARGEST_MAGNITUDE
        NEP_SMALLEST_MAGNITUDE
        NEP_LARGEST_REAL
        NEP_SMALLEST_REAL
        NEP_LARGEST_IMAGINARY
        NEP_SMALLEST_IMAGINARY
        NEP_TARGET_MAGNITUDE
        NEP_TARGET_REAL
        NEP_TARGET_IMAGINARY
        NEP_ALL
        NEP_WHICH_USER

    ctypedef enum SlepcNEPErrorType "NEPErrorType":
        NEP_ERROR_ABSOLUTE
        NEP_ERROR_RELATIVE
        NEP_ERROR_BACKWARD

    ctypedef enum SlepcNEPRefine "NEPRefine":
        NEP_REFINE_NONE
        NEP_REFINE_SIMPLE
        NEP_REFINE_MULTIPLE

    ctypedef enum SlepcNEPRefineScheme "NEPRefineScheme":
        NEP_REFINE_SCHEME_SCHUR
        NEP_REFINE_SCHEME_MBE
        NEP_REFINE_SCHEME_EXPLICIT

    ctypedef enum SlepcNEPConv "NEPConv":
        NEP_CONV_ABS
        NEP_CONV_REL
        NEP_CONV_NORM
        NEP_CONV_USER

    ctypedef enum SlepcNEPStop "NEPStop":
        NEP_STOP_BASIC
        NEP_STOP_USER

    ctypedef enum SlepcNEPConvergedReason "NEPConvergedReason":
        NEP_CONVERGED_TOL
        NEP_CONVERGED_USER
        NEP_DIVERGED_ITS
        NEP_DIVERGED_BREAKDOWN
        NEP_DIVERGED_LINEAR_SOLVE
        NEP_DIVERGED_SUBSPACE_EXHAUSTED
        NEP_CONVERGED_ITERATING

    ctypedef int (*SlepcNEPFunction)(SlepcNEP,
                                     PetscScalar,
                                     PetscMat,
                                     PetscMat,
                                     void*) except PETSC_ERR_PYTHON

    ctypedef int (*SlepcNEPJacobian)(SlepcNEP,
                                     PetscScalar,
                                     PetscMat,
                                     void*) except PETSC_ERR_PYTHON

    ctypedef int (*SlepcNEPCtxDel)(void*)
    ctypedef int (*SlepcNEPStoppingFunction)(SlepcNEP,
                                             PetscInt,
                                             PetscInt,
                                             PetscInt,
                                             PetscInt,
                                             SlepcNEPConvergedReason*,
                                             void*) except PETSC_ERR_PYTHON
    ctypedef int (*SlepcNEPMonitorFunction)(SlepcNEP,
                                            PetscInt,
                                            PetscInt,
                                            PetscScalar*,
                                            PetscScalar*,
                                            PetscReal*,
                                            PetscInt,
                                            void*) except PETSC_ERR_PYTHON

    int NEPCreate(MPI_Comm,SlepcNEP*)
    int NEPDestroy(SlepcNEP*)
    int NEPReset(SlepcNEP)
    int NEPView(SlepcNEP,PetscViewer)

    int NEPSetType(SlepcNEP,SlepcNEPType)
    int NEPGetType(SlepcNEP,SlepcNEPType*)
    int NEPSetTarget(SlepcNEP,PetscScalar)
    int NEPGetTarget(SlepcNEP,PetscScalar*)
    int NEPSetOptionsPrefix(SlepcNEP,char*)
    int NEPGetOptionsPrefix(SlepcNEP,char*[])
    int NEPSetFromOptions(SlepcNEP)
    int NEPAppendOptionsPrefix(SlepcNEP,char*)
    int NEPSetUp(SlepcNEP)
    int NEPSolve(SlepcNEP)

    int NEPSetFunction(SlepcNEP,PetscMat,PetscMat,SlepcNEPFunction,void*)
    int NEPGetFunction(SlepcNEP,PetscMat*,PetscMat*,SlepcNEPFunction*,void**)
    int NEPSetJacobian(SlepcNEP,PetscMat,SlepcNEPJacobian,void*)
    int NEPGetJacobian(SlepcNEP,PetscMat*,SlepcNEPJacobian*,void**)
    int NEPSetSplitOperator(SlepcNEP,PetscInt,PetscMat[],SlepcFN[],PetscMatStructure)
    int NEPGetSplitOperatorTerm(SlepcNEP,PetscInt,PetscMat*,SlepcFN*)
    int NEPGetSplitOperatorInfo(SlepcNEP,PetscInt*,PetscMatStructure*)
    int NEPSetSplitPreconditioner(SlepcNEP,PetscInt,PetscMat[],PetscMatStructure)
    int NEPGetSplitPreconditionerTerm(SlepcNEP,PetscInt,PetscMat*)
    int NEPGetSplitPreconditionerInfo(SlepcNEP,PetscInt*,PetscMatStructure*)

    int NEPSetBV(SlepcNEP,SlepcBV)
    int NEPGetBV(SlepcNEP,SlepcBV*)
    int NEPSetRG(SlepcNEP,SlepcRG)
    int NEPGetRG(SlepcNEP,SlepcRG*)
    int NEPSetDS(SlepcNEP,SlepcDS)
    int NEPGetDS(SlepcNEP,SlepcDS*)
    int NEPSetTolerances(SlepcNEP,PetscReal,PetscInt)
    int NEPGetTolerances(SlepcNEP,PetscReal*,PetscInt*)

    int NEPSetTwoSided(SlepcNEP,PetscBool)
    int NEPGetTwoSided(SlepcNEP,PetscBool*)
    int NEPApplyResolvent(SlepcNEP,SlepcRG,PetscScalar,PetscVec,PetscVec)

    int NEPSetTrackAll(SlepcNEP,PetscBool)
    int NEPGetTrackAll(SlepcNEP,PetscBool*)

    int NEPSetDimensions(SlepcNEP,PetscInt,PetscInt,PetscInt)
    int NEPGetDimensions(SlepcNEP,PetscInt*,PetscInt*,PetscInt*)

    int NEPGetConverged(SlepcNEP,PetscInt*)
    int NEPGetEigenpair(SlepcNEP,PetscInt,PetscScalar*,PetscScalar*,PetscVec,PetscVec)
    int NEPGetLeftEigenvector(SlepcNEP,PetscInt,PetscVec,PetscVec)
    int NEPComputeError(SlepcNEP,PetscInt,SlepcNEPErrorType,PetscReal*)
    int NEPErrorView(SlepcNEP,SlepcNEPErrorType,PetscViewer)
    int NEPValuesView(SlepcNEP,PetscViewer)
    int NEPVectorsView(SlepcNEP,PetscViewer)
    int NEPGetErrorEstimate(SlepcNEP,PetscInt,PetscReal*)

    int NEPMonitorSet(SlepcNEP,SlepcNEPMonitorFunction,void*,SlepcNEPCtxDel)
    int NEPMonitorCancel(SlepcNEP)
    int NEPGetIterationNumber(SlepcNEP,PetscInt*)

    int NEPSetInitialSpace(SlepcNEP,PetscInt,PetscVec*)
    int NEPSetProblemType(SlepcNEP,SlepcNEPProblemType)
    int NEPGetProblemType(SlepcNEP,SlepcNEPProblemType*)
    int NEPSetWhichEigenpairs(SlepcNEP,SlepcNEPWhich)
    int NEPGetWhichEigenpairs(SlepcNEP,SlepcNEPWhich*)

    int NEPSetRefine(SlepcNEP,SlepcNEPRefine,PetscInt,PetscReal,PetscInt,SlepcNEPRefineScheme)
    int NEPGetRefine(SlepcNEP,SlepcNEPRefine*,PetscInt*,PetscReal*,PetscInt*,SlepcNEPRefineScheme*)
    int NEPRefineGetKSP(SlepcNEP,PetscKSP*)

    int NEPGetConvergedReason(SlepcNEP,SlepcNEPConvergedReason*)
    int NEPSetConvergenceTest(SlepcNEP,SlepcNEPConv)
    int NEPGetConvergenceTest(SlepcNEP,SlepcNEPConv*)

    int NEPSetStoppingTestFunction(SlepcNEP,SlepcNEPStoppingFunction,void*,SlepcNEPCtxDel)
    int NEPStoppingBasic(SlepcNEP,PetscInt,PetscInt,PetscInt,PetscInt,SlepcNEPConvergedReason*,void*) except PETSC_ERR_PYTHON

    int NEPRIISetLagPreconditioner(SlepcNEP,PetscInt)
    int NEPRIIGetLagPreconditioner(SlepcNEP,PetscInt*)
    int NEPRIISetConstCorrectionTol(SlepcNEP,PetscBool)
    int NEPRIIGetConstCorrectionTol(SlepcNEP,PetscBool*)
    int NEPRIISetMaximumIterations(SlepcNEP,PetscInt)
    int NEPRIIGetMaximumIterations(SlepcNEP,PetscInt*)
    int NEPRIISetHermitian(SlepcNEP,PetscBool)
    int NEPRIIGetHermitian(SlepcNEP,PetscBool*)
    int NEPRIISetDeflationThreshold(SlepcNEP,PetscReal)
    int NEPRIIGetDeflationThreshold(SlepcNEP,PetscReal*)
    int NEPRIISetKSP(SlepcNEP,PetscKSP)
    int NEPRIIGetKSP(SlepcNEP,PetscKSP*)

    int NEPSLPSetDeflationThreshold(SlepcNEP,PetscReal)
    int NEPSLPGetDeflationThreshold(SlepcNEP,PetscReal*)
    int NEPSLPSetEPS(SlepcNEP,SlepcEPS)
    int NEPSLPGetEPS(SlepcNEP,SlepcEPS*)
    int NEPSLPSetEPSLeft(SlepcNEP,SlepcEPS)
    int NEPSLPGetEPSLeft(SlepcNEP,SlepcEPS*)
    int NEPSLPSetKSP(SlepcNEP,PetscKSP)
    int NEPSLPGetKSP(SlepcNEP,PetscKSP*)

    int NEPNArnoldiSetKSP(SlepcNEP,PetscKSP)
    int NEPNArnoldiGetKSP(SlepcNEP,PetscKSP*)
    int NEPNArnoldiSetLagPreconditioner(SlepcNEP,PetscInt)
    int NEPNArnoldiGetLagPreconditioner(SlepcNEP,PetscInt*)

    int NEPInterpolSetPEP(SlepcNEP,SlepcPEP)
    int NEPInterpolGetPEP(SlepcNEP,SlepcPEP*)
    int NEPInterpolSetInterpolation(SlepcNEP,PetscReal,PetscInt)
    int NEPInterpolGetInterpolation(SlepcNEP,PetscReal*,PetscInt*)

    int NEPNLEIGSSetRestart(SlepcNEP,PetscReal)
    int NEPNLEIGSGetRestart(SlepcNEP,PetscReal*)
    int NEPNLEIGSSetLocking(SlepcNEP,PetscBool)
    int NEPNLEIGSGetLocking(SlepcNEP,PetscBool*)
    int NEPNLEIGSSetInterpolation(SlepcNEP,PetscReal,PetscInt)
    int NEPNLEIGSGetInterpolation(SlepcNEP,PetscReal*,PetscInt*)
    int NEPNLEIGSSetRKShifts(SlepcNEP,PetscInt,PetscScalar[])
    int NEPNLEIGSGetRKShifts(SlepcNEP,PetscInt*,PetscScalar*[])
    int NEPNLEIGSGetKSPs(SlepcNEP,PetscInt*,PetscKSP**)
    int NEPNLEIGSSetFullBasis(SlepcNEP,PetscBool)
    int NEPNLEIGSGetFullBasis(SlepcNEP,PetscBool*)
    int NEPNLEIGSSetEPS(SlepcNEP,SlepcEPS)
    int NEPNLEIGSGetEPS(SlepcNEP,SlepcEPS*)

    ctypedef enum SlepcNEPCISSExtraction "NEPCISSExtraction":
        NEP_CISS_EXTRACTION_RITZ
        NEP_CISS_EXTRACTION_HANKEL
        NEP_CISS_EXTRACTION_CAA

    int NEPCISSSetExtraction(SlepcNEP,SlepcNEPCISSExtraction)
    int NEPCISSGetExtraction(SlepcNEP,SlepcNEPCISSExtraction*)
    int NEPCISSSetSizes(SlepcNEP,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,PetscBool)
    int NEPCISSGetSizes(SlepcNEP,PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscBool*)
    int NEPCISSSetThreshold(SlepcNEP,PetscReal,PetscReal)
    int NEPCISSGetThreshold(SlepcNEP,PetscReal*,PetscReal*)
    int NEPCISSSetRefinement(SlepcNEP,PetscInt,PetscInt)
    int NEPCISSGetRefinement(SlepcNEP,PetscInt*,PetscInt*)
    int NEPCISSGetKSPs(SlepcNEP,PetscInt*,PetscKSP**)

# -----------------------------------------------------------------------------

cdef inline NEP ref_NEP(SlepcNEP nep):
    cdef NEP ob = <NEP> NEP()
    ob.nep = nep
    PetscINCREF(ob.obj)
    return ob

# -----------------------------------------------------------------------------

cdef int NEP_Function(
    SlepcNEP    nep,
    PetscScalar mu,
    PetscMat    A,
    PetscMat    B,
    void*       ctx,
    ) except PETSC_ERR_PYTHON with gil:
    cdef NEP Nep  = ref_NEP(nep)
    cdef Mat Amat = ref_Mat(A)
    cdef Mat Bmat = ref_Mat(B)
    (function, args, kargs) = Nep.get_attr('__function__')
    retv = function(Nep, toScalar(mu), Amat, Bmat, *args, **kargs)
    cdef PetscMat Atmp = NULL, Btmp = NULL
    Atmp = A; A = Amat.mat; Amat.mat = Atmp
    Btmp = B; B = Bmat.mat; Bmat.mat = Btmp
    return 0

# -----------------------------------------------------------------------------

cdef int NEP_Jacobian(
    SlepcNEP    nep,
    PetscScalar mu,
    PetscMat    J,
    void*       ctx,
    ) except PETSC_ERR_PYTHON with gil:
    cdef NEP Nep  = ref_NEP(nep)
    cdef Mat Jmat = ref_Mat(J)
    (jacobian, args, kargs) = Nep.get_attr('__jacobian__')
    retv = jacobian(Nep, toScalar(mu), Jmat, *args, **kargs)
    cdef PetscMat Jtmp = NULL
    Jtmp = J; J = Jmat.mat; Jmat.mat = Jtmp
    return 0

# -----------------------------------------------------------------------------

cdef int NEP_Stopping(
    SlepcNEP                nep,
    PetscInt                its,
    PetscInt                max_it,
    PetscInt                nconv,
    PetscInt                nev,
    SlepcNEPConvergedReason *r,
    void                    *ctx,
    ) except PETSC_ERR_PYTHON with gil:
    cdef NEP Nep = ref_NEP(nep)
    (stopping, args, kargs) = Nep.get_attr('__stopping__')
    reason = stopping(Nep, toInt(its), toInt(max_it), toInt(nconv), toInt(nev), *args, **kargs)
    if   reason is None:  r[0] = NEP_CONVERGED_ITERATING
    elif reason is False: r[0] = NEP_CONVERGED_ITERATING
    elif reason is True:  r[0] = NEP_CONVERGED_USER
    else:                 r[0] = reason

# -----------------------------------------------------------------------------

cdef int NEP_Monitor(
    SlepcNEP    nep,
    PetscInt    its,
    PetscInt    nconv,
    PetscScalar *eigr,
    PetscScalar *eigi,
    PetscReal   *errest,
    PetscInt    nest,
    void        *ctx,
    ) except PETSC_ERR_PYTHON with gil:
    cdef NEP Nep = ref_NEP(nep)
    cdef object monitorlist = Nep.get_attr('__monitor__')
    if monitorlist is None: return 0
    cdef object eig = [toComplex(eigr[i], eigi[i]) for i in range(nest)]
    cdef object err = [toReal(errest[i]) for i in range(nest)]
    for (monitor, args, kargs) in monitorlist:
        monitor(Nep, toInt(its), toInt(nconv), eig, err, *args, **kargs)
    return 0

# -----------------------------------------------------------------------------
