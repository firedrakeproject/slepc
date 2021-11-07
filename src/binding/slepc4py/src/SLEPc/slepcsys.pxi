cdef extern from * :
    enum: PETSC_DECIDE
    enum: PETSC_DEFAULT
    enum: PETSC_DETERMINE

    ctypedef enum PetscBool:
        PETSC_TRUE,  PETSC_YES,
        PETSC_FALSE, PETSC_NO,

    ctypedef enum  PetscNormType "NormType":
        PETSC_NORM_1          "NORM_1"
        PETSC_NORM_2          "NORM_2"
        PETSC_NORM_1_AND_2    "NORM_1_AND_2"
        PETSC_NORM_FROBENIUS  "NORM_FROBENIUS"
        PETSC_NORM_INFINITY   "NORM_INFINITY"
        PETSC_NORM_MAX        "NORM_MAX"

    ctypedef enum  PetscMatStructure "MatStructure":
        MAT_SAME_NONZERO_PATTERN      "SAME_NONZERO_PATTERN"
        MAT_DIFFERENT_NONZERO_PATTERN "DIFFERENT_NONZERO_PATTERN"
        MAT_SUBSET_NONZERO_PATTERN    "SUBSET_NONZERO_PATTERN"

cdef extern from * nogil:
    int PetscMalloc(size_t,void*)
    int PetscFree(void*)
    int PetscMemcpy(void*,void*,size_t)
    int PetscMemzero(void*,size_t)

cdef extern from * nogil:
    MPI_Comm PetscObjectComm(PetscObject)
    int PetscObjectReference(PetscObject)
    int PetscObjectDereference(PetscObject)
    int PetscObjectDestroy(PetscObject*)
    int PetscObjectTypeCompare(PetscObject,char[],PetscBool*)

cdef extern from * nogil:
    int VecCopy(PetscVec,PetscVec)
    int VecSet(PetscVec,PetscScalar)
    int VecDestroy(PetscVec*)

cdef extern from * nogil:
    int MatGetSize(PetscMat,PetscInt*,PetscInt*)
    int MatGetLocalSize(PetscMat,PetscInt*,PetscInt*)

cdef extern from * nogil:
    const_char SLEPC_AUTHOR_INFO[]
    int SlepcGetVersion(char[],size_t)
    int SlepcGetVersionNumber(PetscInt*,PetscInt*,PetscInt*,PetscInt*)

    int SlepcInitialize(int*,char***,char[],char[])
    int SlepcFinalize()
    PetscBool SlepcInitializeCalled
    PetscBool SlepcFinalizeCalled

    int SlepcHasExternalPackage(const char[],PetscBool*)

cdef inline PetscMatStructure matstructure(object structure) \
    except <PetscMatStructure>(-1):
    if   structure is None:  return MAT_DIFFERENT_NONZERO_PATTERN
    elif structure is False: return MAT_DIFFERENT_NONZERO_PATTERN
    elif structure is True:  return MAT_SAME_NONZERO_PATTERN
    else:                    return structure

cdef inline int PetscINCREF(PetscObject *obj):
    if obj    == NULL: return 0
    if obj[0] == NULL: return 0
    return PetscObjectReference(obj[0])

cdef inline int SlepcCLEAR(PetscObject* obj):
    if obj    == NULL: return 0
    if obj[0] == NULL: return 0
    cdef PetscObject tmp
    tmp = obj[0]; obj[0] = NULL
    return PetscObjectDestroy(&tmp)

cdef inline PetscViewer def_Viewer(Viewer viewer):
   return viewer.vwr if viewer is not None else <PetscViewer>NULL

cdef inline KSP ref_KSP(PetscKSP ksp):
    cdef KSP ob = <KSP> KSP()
    ob.ksp = ksp
    PetscINCREF(ob.obj)
    return ob

cdef inline Mat ref_Mat(PetscMat mat):
    cdef Mat ob = <Mat> Mat()
    ob.mat = mat
    PetscINCREF(ob.obj)
    return ob

cdef inline Vec ref_Vec(PetscVec vec):
    cdef Vec ob = <Vec> Vec()
    ob.vec = vec
    PetscINCREF(ob.obj)
    return ob

