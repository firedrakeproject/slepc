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

#if defined(PETSC_BLASLAPACK_UNDERSCORE)

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

#define LAPACKlaev2_ SLEPC_BLASLAPACK(laev2,LAEV2)
#define LAPACKgehrd_ SLEPC_BLASLAPACK(gehrd,GEHRD)
#define LAPACKlanhs_ SLEPC_BLASLAPACK(lanhs,LANHS)
#define LAPACKlange_ SLEPC_BLASLAPACK(lange,LANGE)
#define LAPACKgetri_ SLEPC_BLASLAPACK(getri,GETRI)
#define LAPACKhseqr_ SLEPC_BLASLAPACK(hseqr,HSEQR)
#define LAPACKtrexc_ SLEPC_BLASLAPACK(trexc,TREXC)
#define LAPACKtrevc_ SLEPC_BLASLAPACK(trevc,TREVC)
#define LAPACKsteqr_ SLEPC_BLASLAPACK(steqr,STEQR)
#define LAPACKgeevx_ SLEPC_BLASLAPACK(geevx,GEEVX)
#define LAPACKggevx_ SLEPC_BLASLAPACK(ggevx,GEEVX)

#if !defined(PETSC_USE_COMPLEX)
#define LAPACKorghr_ SLEPC_BLASLAPACK(orghr,ORGHR)
#define LAPACKsyevr_ SLEPC_BLASLAPACK(syevr,SYEVR)
#define LAPACKsygvd_ SLEPC_BLASLAPACK(sygvd,SYGVD)
#else
#define LAPACKorghr_ SLEPC_BLASLAPACK(unghr,UNGHR)
#define LAPACKsyevr_ SLEPC_BLASLAPACK(heevr,HEEVR)
#define LAPACKsygvd_ SLEPC_BLASLAPACK(hegvd,HEGVD)
#endif

EXTERN_C_BEGIN

EXTERN void      LAPACKlaev2_(PetscScalar*,PetscScalar*,PetscScalar*,PetscReal*,PetscReal*,PetscReal*,PetscScalar*);
EXTERN void      LAPACKgehrd_(PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
EXTERN void      LAPACKorghr_(PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
EXTERN PetscReal LAPACKlanhs_(const char*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscBLASInt);
EXTERN PetscReal LAPACKlange_(const char*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscBLASInt);
EXTERN void      LAPACKgetri_(PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
EXTERN void      LAPACKstegr_(const char*,const char*,PetscBLASInt*,PetscReal*,PetscReal*,PetscReal*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscReal*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt,PetscBLASInt);
EXTERN void      LAPACKsteqr_(const char*,PetscBLASInt*,PetscReal*,PetscReal*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscBLASInt);

#if !defined(PETSC_USE_COMPLEX)
EXTERN void      LAPACKhseqr_(const char*,const char*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt,PetscBLASInt);
EXTERN void      LAPACKtrexc_(const char*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt);
EXTERN void      LAPACKtrevc_(const char*,const char*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt,PetscBLASInt);
EXTERN void      LAPACKgeevx_(const char*,const char*,const char*,const char*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt,PetscBLASInt,PetscBLASInt,PetscBLASInt);
EXTERN void      LAPACKggevx_(const char*,const char*,const char*,const char*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt,PetscBLASInt,PetscBLASInt,PetscBLASInt);
EXTERN void      LAPACKsyevr_(const char*,const char*,const char*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt,PetscBLASInt,PetscBLASInt);        
EXTERN void      LAPACKsygvd_(PetscBLASInt*,const char*,const char*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt,PetscBLASInt);
#else
EXTERN void      LAPACKhseqr_(const char*,const char*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt,PetscBLASInt);
EXTERN void      LAPACKtrexc_(const char*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt);
EXTERN void      LAPACKtrevc_(const char*,const char*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscReal*,PetscBLASInt*,PetscBLASInt,PetscBLASInt);
EXTERN void      LAPACKgeevx_(const char*,const char*,const char*,const char*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscReal*,PetscReal*,PetscReal*,PetscReal*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscBLASInt,PetscBLASInt,PetscBLASInt,PetscBLASInt);
EXTERN void      LAPACKggevx_(const char*,const char*,const char*,const char*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*, PetscBLASInt*,PetscReal*,PetscReal*,PetscReal*,PetscReal*,PetscReal*,PetscReal*,PetscScalar*, PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt,PetscBLASInt,PetscBLASInt,PetscBLASInt);
EXTERN void      LAPACKsyevr_(const char *,const char*,const char*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscReal*,PetscScalar*,PetscBLASInt*,PetscBLASInt*, PetscScalar*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt,PetscBLASInt,PetscBLASInt);
EXTERN void      LAPACKsygvd_(PetscBLASInt*,const char*,const char*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt,PetscBLASInt);
#endif

EXTERN_C_END

PETSC_EXTERN_CXX_END
#endif
