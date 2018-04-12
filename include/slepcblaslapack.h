/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2018, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/
/*
   Necessary routines in BLAS and LAPACK not included in petscblaslapack.h
*/

#if !defined(__SLEPCBLASLAPACK_H)
#define __SLEPCBLASLAPACK_H
#include <petscblaslapack.h>

/* Macro to check nonzero info after LAPACK call */
#define SlepcCheckLapackInfo(routine,info) \
  do { \
    if (info) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in LAPACK subroutine %s: info=%d",routine,(int)info); \
  } while (0)

/* LAPACK return type: we assume slange, etc. behave in the same way as snrm2 */
#if defined(PETSC_USE_REAL_SINGLE) && defined(PETSC_BLASLAPACK_SNRM2_RETURNS_DOUBLE)
#define SlepcLRT double
#else
#define SlepcLRT PetscReal
#endif

/* Special macro for srot, csrot, drot, zdrot (BLASMIXEDrot_) */
#if !defined(PETSC_USE_COMPLEX)
# define PETSC_BLASLAPACK_MIXEDPREFIX_ PETSC_BLASLAPACK_PREFIX_
#else
# if defined(PETSC_BLASLAPACK_CAPS)
#  if defined(PETSC_USE_REAL_SINGLE)
#   define PETSC_BLASLAPACK_MIXEDPREFIX_ CS
#  elif defined(PETSC_USE_REAL_DOUBLE)
#   define PETSC_BLASLAPACK_MIXEDPREFIX_ ZD
#  elif defined(PETSC_USE_REAL___FLOAT128)
#   define PETSC_BLASLAPACK_MIXEDPREFIX_ WQ
#  else
#   define PETSC_BLASLAPACK_MIXEDPREFIX_ KH
#  endif
# else
#  if defined(PETSC_USE_REAL_SINGLE)
#   define PETSC_BLASLAPACK_MIXEDPREFIX_ cs
#  elif defined(PETSC_USE_REAL_DOUBLE)
#   define PETSC_BLASLAPACK_MIXEDPREFIX_ zd
#  elif defined(PETSC_USE_REAL___FLOAT128)
#   define PETSC_BLASLAPACK_MIXEDPREFIX_ wq
#  else
#   define PETSC_BLASLAPACK_MIXEDPREFIX_ kh
#  endif
# endif
#endif
#if defined(PETSC_BLASLAPACK_CAPS)
#  define PETSCBLASMIXED(x,X) PETSC_PASTE3(PETSC_BLASLAPACK_MIXEDPREFIX_, X, PETSC_BLASLAPACK_SUFFIX_)
#else
#  define PETSCBLASMIXED(x,X) PETSC_PASTE3(PETSC_BLASLAPACK_MIXEDPREFIX_, x, PETSC_BLASLAPACK_SUFFIX_)
#endif

#if defined(PETSC_BLASLAPACK_STDCALL)
#include <slepcblaslapack_stdcall.h>
#else

#include <slepcblaslapack_mangle.h>

/* LAPACK functions without string parameters */
BLAS_EXTERN void     BLASrot_(PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscScalar*);
BLAS_EXTERN void     BLASMIXEDrot_(PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscReal*);
BLAS_EXTERN void     LAPACKlaev2_(PetscScalar*,PetscScalar*,PetscScalar*,PetscReal*,PetscReal*,PetscReal*,PetscScalar*);
BLAS_EXTERN void     LAPACKgehrd_(PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
BLAS_EXTERN void     LAPACKgelqf_(PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
BLAS_EXTERN void     LAPACKlarfg_(PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*);
BLAS_EXTERN void     LAPACKlag2_(PetscReal*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscReal*,PetscReal*,PetscReal*,PetscReal*,PetscReal*,PetscReal*);
BLAS_EXTERN void     LAPACKlasv2_(PetscReal*,PetscReal*,PetscReal*,PetscReal*,PetscReal*,PetscReal*,PetscReal*,PetscReal*,PetscReal*);
BLAS_EXTERN void     LAPACKlartg_(PetscScalar*,PetscScalar*,PetscReal*,PetscScalar*,PetscScalar*);
BLAS_EXTERN void     LAPACKREALlartg_(PetscReal*,PetscReal*,PetscReal*,PetscReal*,PetscReal*);
BLAS_EXTERN void     LAPACKlaln2_(PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscReal*,PetscReal*,PetscReal*,PetscBLASInt*,PetscReal*,PetscReal*,PetscReal*,PetscBLASInt*,PetscReal*,PetscReal*,PetscReal*,PetscBLASInt*,PetscReal*,PetscReal*,PetscBLASInt*);
BLAS_EXTERN void     LAPACKlaed4_(PetscBLASInt*,PetscBLASInt*,PetscReal*,PetscReal*,PetscReal*,PetscReal*,PetscReal*,PetscBLASInt*);
BLAS_EXTERN void     LAPACKlamrg_(PetscBLASInt*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*);
BLAS_EXTERN SlepcLRT LAPACKlapy2_(PetscReal*,PetscReal*);
BLAS_EXTERN void     LAPACKorghr_(PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
#if !defined(PETSC_USE_COMPLEX)
BLAS_EXTERN void     LAPACKtgexc_(PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
BLAS_EXTERN void     LAPACKgeqp3_(PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
#else
BLAS_EXTERN void     LAPACKtgexc_(PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*);
BLAS_EXTERN void     LAPACKgeqp3_(PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscBLASInt*);
#endif

/* LAPACK functions with string parameters */

/* same name for real and complex */
BLAS_EXTERN void     BLAStrmm_(const char*,const char*,const char*,const char*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*);
BLAS_EXTERN SlepcLRT LAPACKlanhs_(const char*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*);
BLAS_EXTERN SlepcLRT LAPACKlange_(const char*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*);
BLAS_EXTERN SlepcLRT LAPACKpbtrf_(const char*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
BLAS_EXTERN void     LAPACKlarf_(const char*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*);
BLAS_EXTERN void     LAPACKlacpy_(const char*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*);
BLAS_EXTERN SlepcLRT LAPACKlansy_(const char*,const char*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*);
BLAS_EXTERN void     LAPACKlaset_(const char*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscScalar*,PetscBLASInt*);
BLAS_EXTERN void     LAPACKtrsyl_(const char*,const char*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscBLASInt*);
BLAS_EXTERN void     LAPACKtrtri_(const char*,const char*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);

/* subroutines in which we use only the real version, do not care whether they have different name */
BLAS_EXTERN void     LAPACKstevr_(const char*,const char*,PetscBLASInt*,PetscReal*,PetscReal*,PetscReal*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscReal*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*);
BLAS_EXTERN void     LAPACKbdsdc_(const char*,const char*,PetscBLASInt*,PetscReal*,PetscReal*,PetscReal*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscBLASInt*);
BLAS_EXTERN SlepcLRT LAPACKlamch_(const char*);
BLAS_EXTERN SlepcLRT LAPACKlamc3_(PetscReal*,PetscReal*);

/* subroutines with different name in real/complex */
BLAS_EXTERN void     LAPACKormlq_(const char*,const char*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
BLAS_EXTERN void     LAPACKorgtr_(const char*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
BLAS_EXTERN void     LAPACKsytrd_(const char*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscReal*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
#if !defined(PETSC_USE_COMPLEX)
BLAS_EXTERN void     LAPACKsyevr_(const char*,const char*,const char*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*);
BLAS_EXTERN void     LAPACKsyevd_(const char*,const char*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*);
BLAS_EXTERN void     LAPACKsygvd_(PetscBLASInt*,const char*,const char*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*);
#else
BLAS_EXTERN void     LAPACKsyevr_(const char*,const char*,const char*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscReal*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*);
BLAS_EXTERN void     LAPACKsyevd_(const char*,const char*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*);
BLAS_EXTERN void     LAPACKsygvd_(PetscBLASInt*,const char*,const char*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*);
#endif

/* subroutines with different signature in real/complex */
#if !defined(PETSC_USE_COMPLEX)
BLAS_EXTERN void     LAPACKggevx_(const char*,const char*,const char*,const char*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*);
BLAS_EXTERN void     LAPACKggev_(const char*,const char*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
BLAS_EXTERN void     LAPACKtrevc_(const char*,const char*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*);
BLAS_EXTERN void     LAPACKgeevx_(const char*,const char*,const char*,const char*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*);
BLAS_EXTERN void     LAPACKgees_(const char*,const char*,PetscBLASInt(*)(PetscReal,PetscReal),PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*);
BLAS_EXTERN void     LAPACKtrexc_(const char*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*);
BLAS_EXTERN void     LAPACKgesdd_(const char*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*);
BLAS_EXTERN void     LAPACKtgevc_(const char*,const char*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*);
BLAS_EXTERN void     LAPACKhsein_(const char*,const char*,const char*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*);
BLAS_EXTERN void     LAPACKstedc_(const char*,PetscBLASInt*,PetscReal*,PetscReal*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*);
BLAS_EXTERN void     LAPACKlascl_(const char*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
#else
BLAS_EXTERN void     LAPACKggevx_(const char*,const char*,const char*,const char*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*, PetscBLASInt*,PetscReal*,PetscReal*,PetscReal*,PetscReal*,PetscReal*,PetscReal*,PetscScalar*, PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*);
BLAS_EXTERN void     LAPACKggev_(const char*,const char*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscBLASInt*);
BLAS_EXTERN void     LAPACKtrevc_(const char*,const char*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscReal*,PetscBLASInt*);
BLAS_EXTERN void     LAPACKgeevx_(const char*,const char*,const char*,const char*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscReal*,PetscReal*,PetscReal*,PetscReal*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscBLASInt*);
BLAS_EXTERN void     LAPACKgees_(const char*,const char*,PetscBLASInt(*)(PetscScalar),PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscBLASInt*);
BLAS_EXTERN void     LAPACKtrexc_(const char*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*);
BLAS_EXTERN void     LAPACKgesdd_(const char*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscBLASInt*);
BLAS_EXTERN void     LAPACKtgevc_(const char*,const char*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscReal*,PetscBLASInt*);
BLAS_EXTERN void     LAPACKhsein_(const char*,const char*,const char*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*);
BLAS_EXTERN void     LAPACKstedc_(const char*,PetscBLASInt*,PetscReal*,PetscReal*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*);
BLAS_EXTERN void     LAPACKlascl_(const char*,PetscBLASInt*,PetscBLASInt*,PetscReal*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
#endif

#if defined(PETSC_HAVE_COMPLEX)
/* complex subroutines to be called with scalar-type=real */
BLAS_EXTERN void BLASCOMPLEXgemm_(const char*,const char*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscComplex*,PetscComplex*,PetscBLASInt*,PetscComplex*,PetscBLASInt*,PetscComplex*,PetscComplex*,PetscBLASInt*);
BLAS_EXTERN void BLASCOMPLEXscal_(const PetscBLASInt*,const PetscComplex*,PetscComplex*,const PetscBLASInt*);
BLAS_EXTERN void LAPACKCOMPLEXgesv_(const PetscBLASInt*,const PetscBLASInt*,PetscComplex*,const PetscBLASInt*,PetscBLASInt*,PetscComplex*,const PetscBLASInt*,PetscBLASInt*);
#endif

#endif

#endif
