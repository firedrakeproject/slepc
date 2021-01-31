#ifndef SLEPC4PY_COMPAT_H
#define SLEPC4PY_COMPAT_H

#if !defined(PETSC_USE_COMPLEX)

#define SlepcPEPJDUnavailable(pep) do { \
    PetscFunctionBegin; \
    SETERRQ1(PetscObjectComm((PetscObject)pep),PETSC_ERR_SUP,"%s() not available with real scalars",PETSC_FUNCTION_NAME); \
    PetscFunctionReturn(PETSC_ERR_SUP);} while (0)

PetscErrorCode PEPJDSetRestart(PEP pep,PETSC_UNUSED PetscReal keep){SlepcPEPJDUnavailable(pep);}
PetscErrorCode PEPJDGetRestart(PEP pep,PETSC_UNUSED PetscReal *keep){SlepcPEPJDUnavailable(pep);}
PetscErrorCode PEPJDSetFix(PEP pep,PETSC_UNUSED PetscReal fix){SlepcPEPJDUnavailable(pep);}
PetscErrorCode PEPJDGetFix(PEP pep,PETSC_UNUSED PetscReal *fix){SlepcPEPJDUnavailable(pep);}

#undef SlepcPEPJDUnavailable

#endif /*PETSC_USE_COMPLEX*/

#endif /*SLEPC4PY_COMPAT_H*/
