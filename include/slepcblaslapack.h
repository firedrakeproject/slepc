/*

  Necessary routines in BLAS and LAPACK not included in petscblaslapack.f

*/
#if !defined(_SLEPCBLASLAPACK_H)
#define _SLEPCBLASLAPACK_H

#include "petscblaslapack.h"

#if !defined(PETSC_USE_COMPLEX)

/*
    These are real case with no character string arguments
*/

#if defined(PETSC_USES_FORTRAN_SINGLE) 
/*
   For these machines we must call the single precision Fortran version
*/
#define LAlapy2_ SLAPY2
#define LAlaev2_ SLAEV2
#define LAgehrd_ SGEHRD
#define LAorghr_ SORGHR
#define LAlanhs_ SLANHS
#define LAlange_ SLANGE
#define LAgetri_ SGETRI
#define LAhseqr_ SHSEQR
#define LAtrexc_ STREXC
#define LAtrevc_ STREVC
#endif

#if defined(PETSC_HAVE_FORTRAN_UNDERSCORE) || defined(PETSC_BLASLAPACK_F2C)
#define LAlapy2_ dlapy2_
#define LAlaev2_ dlaev2_
#define LAgehrd_ dgehrd_
#define LAorghr_ dorghr_
#define LAlanhs_ dlanhs_
#define LAlange_ dlange_
#define LAgetri_ dgetri_
#define LAhseqr_ dhseqr_
#define LAtrexc_ dtrexc_
#define LAtrevc_ dtrevc_
#elif defined(PETSC_HAVE_FORTRAN_CAPS)
#define LAlapy2_ DLAPY2
#define LAlaev2_ DLAEV2
#define LAgehrd_ DGEHRD
#define LAorghr_ DORGHR
#define LAlanhs_ DLANHS
#define LAlange_ DLANGE
#define LAgetri_ DGETRI
#define LAhseqr_ DHSEQR
#define LAtrexc_ DTREXC
#define LAtrevc_ DTREVC
#else
#define LAlapy2_ dlapy2
#define LAlaev2_ dlaev2
#define LAgehrd_ dgehrd
#define LAorghr_ dorghr
#define LAlanhs_ dlanhs
#define LAlange_ dlange
#define LAgetri_ dgetri
#define LAhseqr_ dhseqr
#define LAtrexc_ dtrexc
#define LAtrevc_ dtrevc
#endif

#else
/*
  Complex
*/
#if defined(PETSC_USES_FORTRAN_SINGLE)
#define DLAPY2   DLAPY2
#define ZLAEV2   CLAEV2
#define ZGEHRD   CGEHRD
#define ZUNGHR   CUNGHR
#define ZLANHS   CLANHS
#define ZLANGE   CLANGE
#define ZGETRI   CGETRI
#define ZHSEQR   CHSEQR
#define ZTREXC   CTREXC
#define ZTREVC   CTREVC
#endif

#if defined(PETSC_HAVE_FORTRAN_UNDERSCORE) || defined(PETSC_BLASLAPACK_F2C)
#define LAlapy2_ dlapy2_
#define LAlaev2_ zlaev2_
#define LAgehrd_ zgehrd_
#define LAorghr_ zunghr_
#define LAlanhs_ zlanhs_
#define LAlange_ zlange_
#define LAgetri_ zgetri_
#define LAhseqr_ zhseqr_
#define LAtrexc_ ztrexc_
#define LAtrevc_ ztrevc_
#elif defined(PETSC_HAVE_FORTRAN_CAPS)
#define LAlapy2_ DLAPY2
#define LAlaev2_ ZLAEV2
#define LAgehrd_ ZGEHRD
#define LAorghr_ ZUNGHR
#define LAlanhs_ ZLANHS
#define LAlange_ ZLANGE
#define LAgetri_ ZGETRI
#define LAhseqr_ ZHSEQR
#define LAtrexc_ ZTREXC
#define LAtrevc_ ZTREVC
#else
#define LAlapy2_ dlapy2
#define LAlaev2_ zlaev2
#define LAgehrd_ zgehrd
#define LAorghr_ zunghr
#define LAlanhs_ zlanhs
#define LAlange_ zlange
#define LAgetri_ zgetri
#define LAhseqr_ zhseqr
#define LAtrexc_ ztrexc
#define LAtrevc_ ztrevc
#endif

#endif

EXTERN_C_BEGIN

EXTERN PetscReal LAlapy2_(PetscReal*,PetscReal*);
EXTERN void      LAlaev2_(PetscScalar*,PetscScalar*,PetscScalar*,PetscReal*,PetscReal*,PetscReal*,PetscScalar*);
EXTERN void      LAgehrd_(int*,int*,int*,PetscScalar*,int*,PetscScalar*,PetscScalar*,int*,int*);
EXTERN void      LAorghr_(int*,int*,int*,PetscScalar*,int*,PetscScalar*,PetscScalar*,int*,int*);
EXTERN PetscReal LAlanhs_(const char*,int*,PetscScalar*,int*,PetscReal*,int);
EXTERN PetscReal LAlange_(const char*,int*,int*,PetscScalar*,int*,PetscReal*,int);
EXTERN void      LAgetri_(int*,PetscScalar*,int*,int*,PetscScalar*,int*,int*);

#if !defined(PETSC_USE_COMPLEX)
EXTERN void      LAhseqr_(const char*,const char*,int*,int*,int*,PetscScalar*,int*,PetscScalar*,PetscScalar*,PetscScalar*,int*,PetscScalar*,int*,int*,int,int);
EXTERN void      LAtrexc_(const char*,int*,PetscScalar*,int*,PetscScalar*,int*,int*,int*,PetscScalar*,int*,int);
EXTERN void      LAtrevc_(const char*,const char*,int*,int*,PetscScalar*,int*,PetscScalar*,int*,PetscScalar*,int*,int*,int*,PetscScalar*,int*,int,int);
#else
EXTERN void      LAhseqr_(const char*,const char*,int*,int*,int*,PetscScalar*,int*,PetscScalar*,PetscScalar*,int*,PetscScalar*,int*,int*,int,int);
EXTERN void      LAtrexc_(const char*,int*,PetscScalar*,int*,PetscScalar*,int*,int*,int*,int*,int);
EXTERN void      LAtrevc_(const char*,const char*,int*,int*,PetscScalar*,int*,PetscScalar*,int*,PetscScalar*,int*,int*,int*,PetscScalar*,PetscReal*,int*,int,int);
#endif

EXTERN_C_END

#endif

