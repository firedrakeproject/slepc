/*

  Necessary routines in BLAS and LAPACK not included in petscblaslapack.f

*/
#if !defined(_SLEPCBLASLAPACK_H)
#define _SLEPCBLASLAPACK_H
#include "petscblaslapack.h"
PETSC_EXTERN_CXX_BEGIN

#define SLEPC_CONCAT(a,b) a##b
#define SLEPC_CONCAT3(a,b,c) a##b##c

#if defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define SLEPC_FORTRAN(lcase,ucase) SLEPC_CONCAT(lcase,_)
#elif defined(PETSC_HAVE_FORTRAN_CAPS)
#define SLEPC_FORTRAN(lcase,ucase) ucase
#else
#define SLEPC_FORTRAN(lcase,ucase) lcase
#endif

#if defined(PETSC_BLASLAPACK_F2C)

#if !defined(PETSC_USE_COMPLEX)
/* real numbers */
#if defined(PETSC_USES_FORTRAN_SINGLE) || defined(PETSC_USE_SINGLE)
/* single precision */
#define SLEPC_BLASLAPACK(lcase,ucase) SLEPC_CONCAT3(s,lcase,_)
#else
/* double precision */
#define SLEPC_BLASLAPACK(lcase,ucase) SLEPC_CONCAT3(d,lcase,_)
#endif
#else
/* complex numbers */
#if defined(PETSC_USES_FORTRAN_SINGLE) || defined(PETSC_USE_SINGLE)
/* single precision */
#define SLEPC_BLASLAPACK(lcase,ucase) SLEPC_CONCAT3(c,lcase,_)
#else
/* double precision */
#define SLEPC_BLASLAPACK(lcase,ucase) SLEPC_CONCAT3(z,lcase,_)
#endif
#endif

#else

#if !defined(PETSC_USE_COMPLEX)
/* real numbers */
#if defined(PETSC_USES_FORTRAN_SINGLE) || defined(PETSC_USE_SINGLE)
/* single precision */
#define SLEPC_BLASLAPACK(lcase,ucase) SLEPC_FORTRAN(SLEPC_CONCAT(s,lcase),SLEPC_CONCAT(S,ucase))
#else
/* double precision */
#define SLEPC_BLASLAPACK(lcase,ucase) SLEPC_FORTRAN(SLEPC_CONCAT(d,lcase),SLEPC_CONCAT(D,ucase))
#endif
#else
/* complex numbers */
#if defined(PETSC_USES_FORTRAN_SINGLE) || defined(PETSC_USE_SINGLE)
/* single precision */
#define SLEPC_BLASLAPACK(lcase,ucase) SLEPC_FORTRAN(SLEPC_CONCAT(c,lcase),SLEPC_CONCAT(C,ucase))
#else
/* double precision */
#define SLEPC_BLASLAPACK(lcase,ucase) SLEPC_FORTRAN(SLEPC_CONCAT(z,lcase),SLEPC_CONCAT(Z,ucase))
#endif
#endif

#endif

#define LAlaev2_ SLEPC_BLASLAPACK(laev2,LAEV2)
#define LAgehrd_ SLEPC_BLASLAPACK(gehrd,GEHRD)
#if !defined(PETSC_USE_COMPLEX)
#define LAorghr_ SLEPC_BLASLAPACK(orghr,ORGHR)
#else
#define LAorghr_ SLEPC_BLASLAPACK(unghr,UNGHR)
#endif
#define LAlanhs_ SLEPC_BLASLAPACK(lanhs,LANHS)
#define LAlange_ SLEPC_BLASLAPACK(lange,LANGE)
#define LAgetri_ SLEPC_BLASLAPACK(getri,GETRI)
#define LAhseqr_ SLEPC_BLASLAPACK(hseqr,HSEQR)
#define LAtrexc_ SLEPC_BLASLAPACK(trexc,TREXC)
#define LAtrevc_ SLEPC_BLASLAPACK(trevc,TREVC)

EXTERN_C_BEGIN

EXTERN void      LAlaev2_(PetscScalar*,PetscScalar*,PetscScalar*,PetscReal*,PetscReal*,PetscReal*,PetscScalar*);
EXTERN void      LAgehrd_(PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
EXTERN void      LAorghr_(PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
EXTERN PetscReal LAlanhs_(const char*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscBLASInt);
EXTERN PetscReal LAlange_(const char*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscBLASInt);
EXTERN void      LAgetri_(PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);

#if !defined(PETSC_USE_COMPLEX)
EXTERN void      LAhseqr_(const char*,const char*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt,PetscBLASInt);
EXTERN void      LAtrexc_(const char*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt);
EXTERN void      LAtrevc_(const char*,const char*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt,PetscBLASInt);
#else
EXTERN void      LAhseqr_(const char*,const char*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt,PetscBLASInt);
EXTERN void      LAtrexc_(const char*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt);
EXTERN void      LAtrevc_(const char*,const char*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscReal*,PetscBLASInt*,PetscBLASInt,PetscBLASInt);
#endif

EXTERN_C_END

PETSC_EXTERN_CXX_END
#endif
