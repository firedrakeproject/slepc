cdef extern from * nogil:

    ctypedef char* SlepcSTType "const char*"
    SlepcSTType STSHELL
    SlepcSTType STSHIFT
    SlepcSTType STSINVERT
    SlepcSTType STCAYLEY
    SlepcSTType STPRECOND
    SlepcSTType STFILTER

    ctypedef enum SlepcSTMatMode "STMatMode":
        ST_MATMODE_COPY
        ST_MATMODE_INPLACE
        ST_MATMODE_SHELL

    int STView(SlepcST,PetscViewer)
    int STDestroy(SlepcST*)
    int STReset(SlepcST)
    int STCreate(MPI_Comm,SlepcST*)
    int STGetType(SlepcST,SlepcSTType*)
    int STSetType(SlepcST,SlepcSTType)
    int STGetOptionsPrefix(SlepcST,char*[])
    int STSetOptionsPrefix(SlepcST,char[])
    int STAppendOptionsPrefix(SlepcST,char[])
    int STSetFromOptions(SlepcST)

    int STGetShift(SlepcST,PetscScalar*)
    int STSetShift(SlepcST,PetscScalar)

    int STGetKSP(SlepcST,PetscKSP*)
    int STSetKSP(SlepcST,PetscKSP)

    int STGetOperator(SlepcST,PetscMat*)
    int STRestoreOperator(SlepcST,PetscMat*)

    int STGetNumMatrices(SlepcST,PetscInt*)
    int STGetMatrix(SlepcST,PetscInt,PetscMat*)
    int STSetMatrices(SlepcST,PetscInt,PetscMat*)
    int STSetMatStructure(SlepcST,PetscMatStructure)
    int STGetMatStructure(SlepcST,PetscMatStructure*)
    int STSetPreconditionerMat(SlepcST,PetscMat)
    int STGetPreconditionerMat(SlepcST,PetscMat*)

    int STSetTransform(SlepcST,PetscBool)
    int STGetTransform(SlepcST,PetscBool*)

    int STGetMatMode(SlepcST,SlepcSTMatMode*)
    int STSetMatMode(SlepcST,SlepcSTMatMode)

    int STSetUp(SlepcST)
    int STApply(SlepcST,PetscVec,PetscVec)
    int STApplyMat(SlepcST,PetscMat,PetscMat)
    int STApplyTranspose(SlepcST,PetscVec,PetscVec)
    int STApplyHermitianTranspose(SlepcST,PetscVec,PetscVec)

    int STCayleySetAntishift(SlepcST,PetscScalar)
    int STCayleyGetAntishift(SlepcST,PetscScalar*)

    int STFilterSetInterval(SlepcST,PetscReal,PetscReal)
    int STFilterGetInterval(SlepcST,PetscReal*,PetscReal*)
    int STFilterSetRange(SlepcST,PetscReal,PetscReal)
    int STFilterGetRange(SlepcST,PetscReal*,PetscReal*)
    int STFilterSetDegree(SlepcST,PetscInt)
    int STFilterGetDegree(SlepcST,PetscInt*)

