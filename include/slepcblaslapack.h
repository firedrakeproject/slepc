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
#define DLARNV   SLARNV
#define DLAPY2   SLAPY2
#define DLAEV2   SLAEV2
#define DGELQF   SGELQF
#define DORMLQ   SORMLQ
#define DHSEQR   SHSEQR
#define DTREVC   STREVC
#define DGEHRD   SGEHRD
#define DORGHR   SORGHR
#define DGEES    SGEES 
#endif

#if defined(PETSC_HAVE_FORTRAN_UNDERSCORE) || defined(PETSC_BLASLAPACK_F2C)
#define LArnv_   dlarnv_
#define LAlapy2_ dlapy2_
#define LAlaev2_ dlaev2_
#define LAgelqf_ dgelqf_
#define LAormlq_ dormlq_
#define LAhseqr_ dhseqr_
#define LAtrevc_ dtrevc_
#define LAgehrd_ dgehrd_
#define LAorghr_ dorghr_
#define LAgees_  dgees_
#elif defined(PETSC_HAVE_FORTRAN_CAPS)
#define LArnv_   DLARNV
#define LAlapy2_ DLAPY2
#define LAlaev2_ DLAEV2
#define LAgelqf_ DGELQF
#define LAormlq_ DORMLQ
#define LAhseqr_ DHSEQR
#define LAtrevc_ DTREVC
#define LAgehrd_ DGEHRD
#define LAorghr_ DORGHR
#define LAgees_  DGEES 
#else
#define LArnv_   dlarnv
#define LAlapy2_ dlapy2
#define LAlaev2_ dlaev2
#define LAgelqf_ dgelqf
#define LAormlq_ dormlq
#define LAhseqr_ dhseqr
#define LAtrevc_ dtrevc
#define LAgehrd_ dgehrd
#define LAorghr_ dorghr
#define LAgees_  dgees
#endif

#else
/*
   Complex with no character string arguments
*/
#if defined(PETSC_USES_FORTRAN_SINGLE)
#define ZLARNV   CLARNV
#define ZGELQF   CGELQF
#define ZUNMLQ   CUNMLQ
#define ZHSEQR   CHSEQR
#define ZTREVC   CTREVC
#define ZGEHRD   CGEHRD
#define ZUNGHR   CUNGHR
#define ZGEES    CGEES 
#endif

#if defined(PETSC_HAVE_FORTRAN_UNDERSCORE) || defined(PETSC_BLASLAPACK_F2C)
#define LArnv_   zlarnv_
#define LAlaev2_ zlaev2_
#define LAgelqf_ zgelqf_
#define LAormlq_ zunmlq_
#define LAhseqr_ zhseqr_
#define LAtrevc_ ztrevc_
#define LAgehrd_ zgehrd_
#define LAorghr_ zunghr_
#define LAgees_  zgees_ 
#elif defined(PETSC_HAVE_FORTRAN_CAPS)
#define LArnv_   ZLARNV
#define LAlaev2_ ZLAEV2
#define LAgelqf_ ZGELQF
#define LAormlq_ ZUNMLQ
#define LAhseqr_ ZHSEQR
#define LAtrevc_ ZTREVC
#define LAgehrd_ ZGEHRD
#define LAorghr_ ZUNGHR
#define LAgees_  ZGEES 
#else
#define LArnv_   zlarnv
#define LAlaev2_ zlaev2
#define LAgelqf_ zgelqf
#define LAormlq_ zunmlq
#define LAhseqr_ zhseqr
#define LAtrevc_ ztrevc
#define LAgehrd_ zgehrd
#define LAorghr_ zunghr
#define LAgees_  zgees 
#endif

#endif

EXTERN_C_BEGIN

extern void   LArnv_(int*,int*,int*,PetscScalar*);
extern double LAlapy2_(double*,double*);
extern void   LAlaev2_(PetscScalar*,PetscScalar*,PetscScalar*,PetscReal*,PetscReal*,PetscReal*,PetscScalar*);
extern void   LAgelqf_(int*,int*,PetscScalar*,int*,PetscScalar*,PetscScalar*,int*,int*);
extern void   LAormlq_(char*,char*,int*,int*,int*,PetscScalar*,int*,PetscScalar*,PetscScalar*,int*,PetscScalar*,int*,int*,int,int);
extern void   LAgehrd_(int*,int*,int*,PetscScalar*,int*,PetscScalar*,PetscScalar*,int*,int*);
extern void   LAorghr_(int*,int*,int*,PetscScalar*,int*,PetscScalar*,PetscScalar*,int*,int*);
#if !defined(PETSC_USE_COMPLEX)
extern void   LAhseqr_(char*,char*,int*,int*,int*,PetscScalar*,int*,PetscScalar*,PetscScalar*,PetscScalar*,int*,PetscScalar*,int*,int*,int,int);
extern void   LAgees_(char*,char*,int*,int*,PetscScalar*,int*,int*,PetscScalar*,PetscScalar*,PetscScalar*,int*,PetscScalar*,int*,int*,int*);
extern void   LAtrevc_(char*,char*,int*,int*,PetscScalar*,int*,PetscScalar*,int*,PetscScalar*,int*,int*,int*,PetscScalar*,int*,int,int);
#else
extern void   LAhseqr_(char*,char*,int*,int*,int*,PetscScalar*,int*,PetscScalar*,PetscScalar*,int*,PetscScalar*,int*,int*,int,int);
extern void   LAgees_(char*,char*,int*,int*,PetscScalar*,int*,int*,PetscScalar*,PetscScalar*,int*,PetscScalar*,int*,PetscReal*,int*,int*);
extern void   LAtrevc_(char*,char*,int*,int*,PetscScalar*,int*,PetscScalar*,int*,PetscScalar*,int*,int*,int*,PetscScalar*,PetscReal*,int*,int,int);
#endif

EXTERN_C_END

#endif

