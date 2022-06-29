cdef extern from * nogil:

    ctypedef char* SlepcSVDType "const char*"
    SlepcSVDType SVDCROSS
    SlepcSVDType SVDCYCLIC
    SlepcSVDType SVDLAPACK
    SlepcSVDType SVDLANCZOS
    SlepcSVDType SVDTRLANCZOS
    SlepcSVDType SVDRANDOMIZED
    SlepcSVDType SVDSCALAPACK
    SlepcSVDType SVDELEMENTAL
    SlepcSVDType SVDPRIMME

    ctypedef enum SlepcSVDProblemType "SVDProblemType":
        SVD_STANDARD
        SVD_GENERALIZED
        SVD_HYPERBOLIC

    ctypedef enum SlepcSVDWhich "SVDWhich":
        SVD_LARGEST
        SVD_SMALLEST

    ctypedef enum SlepcSVDErrorType "SVDErrorType":
        SVD_ERROR_ABSOLUTE
        SVD_ERROR_RELATIVE
        SVD_ERROR_NORM

    ctypedef enum SlepcSVDConv "SVDConv":
        SVD_CONV_ABS
        SVD_CONV_REL
        SVD_CONV_NORM
        SVD_CONV_MAXIT
        SVD_CONV_USER

    ctypedef enum SlepcSVDStop "SVDStop":
        SVD_STOP_BASIC
        SVD_STOP_USER

    ctypedef enum SlepcSVDConvergedReason "SVDConvergedReason":
        SVD_CONVERGED_TOL
        SVD_CONVERGED_USER
        SVD_CONVERGED_MAXIT
        SVD_DIVERGED_ITS
        SVD_DIVERGED_BREAKDOWN
        SVD_CONVERGED_ITERATING

    ctypedef int (*SlepcSVDCtxDel)(void*)
    ctypedef int (*SlepcSVDStoppingFunction)(SlepcSVD,
                                             PetscInt,
                                             PetscInt,
                                             PetscInt,
                                             PetscInt,
                                             SlepcSVDConvergedReason*,
                                             void*) except PETSC_ERR_PYTHON
    ctypedef int (*SlepcSVDMonitorFunction)(SlepcSVD,
                                            PetscInt,
                                            PetscInt,
                                            PetscReal*,
                                            PetscReal*,
                                            PetscInt,
                                            void*) except PETSC_ERR_PYTHON

    int SVDCreate(MPI_Comm,SlepcSVD*)
    int SVDView(SlepcSVD,PetscViewer)
    int SVDDestroy(SlepcSVD*)
    int SVDReset(SlepcSVD)
    int SVDSetType(SlepcSVD,SlepcSVDType)
    int SVDGetType(SlepcSVD,SlepcSVDType*)
    int SVDSetOptionsPrefix(SlepcSVD,char[])
    int SVDAppendOptionsPrefix(SlepcSVD,char[])
    int SVDGetOptionsPrefix(SlepcSVD,char*[])
    int SVDSetFromOptions(SlepcSVD)

    int SVDSetProblemType(SlepcSVD,SlepcSVDProblemType)
    int SVDGetProblemType(SlepcSVD,SlepcSVDProblemType*)
    int SVDIsGeneralized(SlepcSVD,PetscBool*)
    int SVDIsHyperbolic(SlepcSVD,PetscBool*)
    int SVDSetBV(SlepcSVD,SlepcBV,SlepcBV)
    int SVDGetBV(SlepcSVD,SlepcBV*,SlepcBV*)
    int SVDSetDS(SlepcSVD,SlepcDS)
    int SVDGetDS(SlepcSVD,SlepcDS*)

    int SVDSetOperators(SlepcSVD,PetscMat,PetscMat)
    int SVDGetOperators(SlepcSVD,PetscMat*,PetscMat*)
    int SVDSetSignature(SlepcSVD,PetscVec)
    int SVDGetSignature(SlepcSVD,PetscVec*)

    int SVDSetInitialSpaces(SlepcSVD,PetscInt,PetscVec*,PetscInt,PetscVec*)

    int SVDSetImplicitTranspose(SlepcSVD,PetscBool)
    int SVDGetImplicitTranspose(SlepcSVD,PetscBool*)
    int SVDSetDimensions(SlepcSVD,PetscInt,PetscInt,PetscInt)
    int SVDGetDimensions(SlepcSVD,PetscInt*,PetscInt*,PetscInt*)
    int SVDSetTolerances(SlepcSVD,PetscReal,PetscInt)
    int SVDGetTolerances(SlepcSVD,PetscReal*,PetscInt*)
    int SVDSetWhichSingularTriplets(SlepcSVD,SlepcSVDWhich)
    int SVDGetWhichSingularTriplets(SlepcSVD,SlepcSVDWhich*)
    int SVDSetConvergenceTest(SlepcSVD,SlepcSVDConv)
    int SVDGetConvergenceTest(SlepcSVD,SlepcSVDConv*)

    int SVDMonitorSet(SlepcSVD,SlepcSVDMonitorFunction,void*,SlepcSVDCtxDel)
    int SVDMonitorCancel(SlepcSVD)

    int SVDSetStoppingTestFunction(SlepcSVD,SlepcSVDStoppingFunction,void*,SlepcSVDCtxDel)
    int SVDStoppingBasic(SlepcSVD,PetscInt,PetscInt,PetscInt,PetscInt,SlepcSVDConvergedReason*,void*) except PETSC_ERR_PYTHON

    int SVDSetTrackAll(SlepcSVD,PetscBool)
    int SVDGetTrackAll(SlepcSVD,PetscBool*)

    int SVDSetUp(SlepcSVD)
    int SVDSolve(SlepcSVD)
    int SVDGetIterationNumber(SlepcSVD,PetscInt*)
    int SVDGetConvergedReason(SlepcSVD,SlepcSVDConvergedReason*)
    int SVDGetConverged(SlepcSVD,PetscInt*)
    int SVDGetSingularTriplet(SlepcSVD,PetscInt,PetscReal*,PetscVec,PetscVec)
    int SVDComputeError(SlepcSVD,PetscInt,SlepcSVDErrorType,PetscReal*)
    int SVDErrorView(SlepcSVD,SlepcSVDErrorType,PetscViewer)
    int SVDValuesView(SlepcSVD,PetscViewer)
    int SVDVectorsView(SlepcSVD,PetscViewer)

    int SVDCrossSetExplicitMatrix(SlepcSVD,PetscBool)
    int SVDCrossGetExplicitMatrix(SlepcSVD,PetscBool*)
    int SVDCrossSetEPS(SlepcSVD,SlepcEPS)
    int SVDCrossGetEPS(SlepcSVD,SlepcEPS*)

    int SVDCyclicSetExplicitMatrix(SlepcSVD,PetscBool)
    int SVDCyclicGetExplicitMatrix(SlepcSVD,PetscBool*)
    int SVDCyclicSetEPS(SlepcSVD,SlepcEPS)
    int SVDCyclicGetEPS(SlepcSVD,SlepcEPS*)

    int SVDLanczosSetOneSide(SlepcSVD,PetscBool)
    int SVDLanczosGetOneSide(SlepcSVD,PetscBool*)

    int SVDTRLanczosSetOneSide(SlepcSVD,PetscBool)
    int SVDTRLanczosGetOneSide(SlepcSVD,PetscBool*)
    int SVDTRLanczosSetGBidiag(SlepcSVD,SlepcSVDTRLanczosGBidiag)
    int SVDTRLanczosGetGBidiag(SlepcSVD,SlepcSVDTRLanczosGBidiag*)
    int SVDTRLanczosSetKSP(SlepcSVD,PetscKSP)
    int SVDTRLanczosGetKSP(SlepcSVD,PetscKSP*)
    int SVDTRLanczosSetRestart(SlepcSVD,PetscReal)
    int SVDTRLanczosGetRestart(SlepcSVD,PetscReal*)
    int SVDTRLanczosSetLocking(SlepcSVD,PetscBool)
    int SVDTRLanczosGetLocking(SlepcSVD,PetscBool*)
    int SVDTRLanczosSetExplicitMatrix(SlepcSVD,PetscBool)
    int SVDTRLanczosGetExplicitMatrix(SlepcSVD,PetscBool*)

    ctypedef enum SlepcSVDTRLanczosGBidiag "SVDTRLanczosGBidiag":
        SVD_TRLANCZOS_GBIDIAG_SINGLE
        SVD_TRLANCZOS_GBIDIAG_UPPER
        SVD_TRLANCZOS_GBIDIAG_LOWER

# -----------------------------------------------------------------------------

cdef inline SVD ref_SVD(SlepcSVD svd):
    cdef SVD ob = <SVD> SVD()
    ob.svd = svd
    PetscINCREF(ob.obj)
    return ob

# -----------------------------------------------------------------------------

cdef int SVD_Stopping(
    SlepcSVD                svd,
    PetscInt                its,
    PetscInt                max_it,
    PetscInt                nconv,
    PetscInt                nev,
    SlepcSVDConvergedReason *r,
    void                    *ctx,
    ) except PETSC_ERR_PYTHON with gil:
    cdef SVD Svd = ref_SVD(svd)
    (stopping, args, kargs) = Svd.get_attr('__stopping__')
    reason = stopping(Svd, toInt(its), toInt(max_it), toInt(nconv), toInt(nev), *args, **kargs)
    if   reason is None:  r[0] = SVD_CONVERGED_ITERATING
    elif reason is False: r[0] = SVD_CONVERGED_ITERATING
    elif reason is True:  r[0] = SVD_CONVERGED_USER
    else:                 r[0] = reason

# -----------------------------------------------------------------------------

cdef int SVD_Monitor(
    SlepcSVD    svd,
    PetscInt    its,
    PetscInt    nconv,
    PetscReal   *sigma,
    PetscReal   *errest,
    PetscInt    nest,
    void        *ctx,
    ) except PETSC_ERR_PYTHON with gil:
    cdef SVD Svd = ref_SVD(svd)
    cdef object monitorlist = Svd.get_attr('__monitor__')
    if monitorlist is None: return 0
    cdef object eig = [toReal(sigma[i]) for i in range(nest)]
    cdef object err = [toReal(errest[i]) for i in range(nest)]
    for (monitor, args, kargs) in monitorlist:
        monitor(Svd, toInt(its), toInt(nconv), eig, err, *args, **kargs)
    return 0

# -----------------------------------------------------------------------------
