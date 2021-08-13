cdef extern from * nogil:

    ctypedef char* SlepcRGType "const char*"
    SlepcRGType RGINTERVAL
    SlepcRGType RGPOLYGON
    SlepcRGType RGELLIPSE
    SlepcRGType RGRING

    ctypedef enum SlepcRGQuadRule "RGQuadRule":
        RG_QUADRULE_TRAPEZOIDAL
        RG_QUADRULE_CHEBYSHEV

    int RGCreate(MPI_Comm,SlepcRG*)
    int RGView(SlepcRG,PetscViewer)
    int RGDestroy(SlepcRG*)
    int RGSetType(SlepcRG,SlepcRGType)
    int RGGetType(SlepcRG,SlepcRGType*)

    int RGSetOptionsPrefix(SlepcRG,char[])
    int RGGetOptionsPrefix(SlepcRG,char*[])
    int RGAppendOptionsPrefix(SlepcRG,char[])
    int RGSetFromOptions(SlepcRG)

    int RGIsTrivial(SlepcRG,PetscBool*)
    int RGIsAxisymmetric(SlepcRG,PetscBool,PetscBool*)
    int RGSetComplement(SlepcRG,PetscBool)
    int RGGetComplement(SlepcRG,PetscBool*)

    int RGSetScale(SlepcRG,PetscReal)
    int RGGetScale(SlepcRG,PetscReal*)
    int RGPushScale(SlepcRG,PetscReal)
    int RGPopScale(SlepcRG)

    int RGCheckInside(SlepcRG,PetscInt,PetscScalar*,PetscScalar*,PetscInt*)
    int RGComputeContour(SlepcRG,PetscInt,PetscScalar*,PetscScalar*)
    int RGComputeBoundingBox(SlepcRG,PetscReal*,PetscReal*,PetscReal*,PetscReal*)
    int RGCanUseConjugates(SlepcRG,PetscBool,PetscBool*)
    int RGComputeQuadrature(SlepcRG,SlepcRGQuadRule,PetscInt,PetscScalar*,PetscScalar*,PetscScalar*)

    int RGEllipseSetParameters(SlepcRG,PetscScalar,PetscReal,PetscReal)
    int RGEllipseGetParameters(SlepcRG,PetscScalar*,PetscReal*,PetscReal*)
    int RGIntervalSetEndpoints(SlepcRG,PetscReal,PetscReal,PetscReal,PetscReal)
    int RGIntervalGetEndpoints(SlepcRG,PetscReal*,PetscReal*,PetscReal*,PetscReal*)
    int RGPolygonSetVertices(SlepcRG,PetscInt,PetscScalar*,PetscScalar*)
    int RGPolygonGetVertices(SlepcRG,PetscInt*,PetscScalar**,PetscScalar**)
    int RGRingSetParameters(SlepcRG,PetscScalar,PetscReal,PetscReal,PetscReal,PetscReal,PetscReal)
    int RGRingGetParameters(SlepcRG,PetscScalar*,PetscReal*,PetscReal*,PetscReal*,PetscReal*,PetscReal*)

