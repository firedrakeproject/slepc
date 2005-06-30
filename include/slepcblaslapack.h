/*

  Necessary routines in BLAS and LAPACK not included in petscblaslapack.f

*/
#if !defined(_SLEPCBLASLAPACK_H)
#define _SLEPCBLASLAPACK_H
#include "petscblaslapack.h"
PETSC_EXTERN_CXX_BEGIN


#if defined(PETSC_HAVE_FORTRAN_UNDERSCORE) || defined(PETSC_BLASLAPACK_UNDERSCORE)
#if defined(PETSC_USE_SINGLE)
#define SLEPC_BLASLAPACKREAL(lcase,ucase) s##lcase##_
#if defined(PETSC_USE_COMPLEX)
#define SLEPC_BLASLAPACK(lcase,ucase) c##lcase##_
#else
#define SLEPC_BLASLAPACK(lcase,ucase) s##lcase##_
#endif
#else
#define SLEPC_BLASLAPACKREAL(lcase,ucase) d##lcase##_
#if defined(PETSC_USE_COMPLEX)
#define SLEPC_BLASLAPACK(lcase,ucase) z##lcase##_
#else
#define SLEPC_BLASLAPACK(lcase,ucase) d##lcase##_
#endif
#endif

#elif defined(PETSC_HAVE_FORTRAN_CAPS)
#if defined(PETSC_USE_SINGLE)
#define SLEPC_BLASLAPACKREAL(lcase,ucase) S##ucase
#if defined(PETSC_USE_COMPLEX)
#define SLEPC_BLASLAPACK(lcase,ucase) C##ucase
#else
#define SLEPC_BLASLAPACK(lcase,ucase) S##ucase
#endif
#else
#define SLEPC_BLASLAPACKREAL(lcase,ucase) D##ucase
#if defined(PETSC_USE_COMPLEX)
#define SLEPC_BLASLAPACK(lcase,ucase) Z##ucase
#else
#define SLEPC_BLASLAPACK(lcase,ucase) D##ucase
#endif
#endif

#else
#if defined(PETSC_USE_SINGLE)
#define SLEPC_BLASLAPACKREAL(lcase,ucase) s##lcase
#if defined(PETSC_USE_COMPLEX)
#define SLEPC_BLASLAPACK(lcase,ucase) c##lcase
#else
#define SLEPC_BLASLAPACK(lcase,ucase) s##lcase
#endif
#else
#define SLEPC_BLASLAPACKREAL(lcase,ucase) d##lcase
#if defined(PETSC_USE_COMPLEX)
#define SLEPC_BLASLAPACK(lcase,ucase) z##lcase
#else
#define SLEPC_BLASLAPACK(lcase,ucase) d##lcase
#endif
#endif

#endif

#define BLAStrsm_    SLEPC_BLASLAPACK(trsm,TRSM)
#define LAPACKlaev2_ SLEPC_BLASLAPACK(laev2,LAEV2)
#define LAPACKgehrd_ SLEPC_BLASLAPACK(gehrd,GEHRD)
#define LAPACKlanhs_ SLEPC_BLASLAPACK(lanhs,LANHS)
#define LAPACKlange_ SLEPC_BLASLAPACK(lange,LANGE)
#define LAPACKgetri_ SLEPC_BLASLAPACK(getri,GETRI)
#define LAPACKhseqr_ SLEPC_BLASLAPACK(hseqr,HSEQR)
#define LAPACKtrexc_ SLEPC_BLASLAPACK(trexc,TREXC)
#define LAPACKtrevc_ SLEPC_BLASLAPACK(trevc,TREVC)
#define LAPACKgeevx_ SLEPC_BLASLAPACK(geevx,GEEVX)
#define LAPACKggevx_ SLEPC_BLASLAPACK(ggevx,GEEVX)
#define LAPACKgelqf_ SLEPC_BLASLAPACK(gelqf,GELQF)

#if !defined(PETSC_USE_COMPLEX)
#define LAPACKorghr_ SLEPC_BLASLAPACK(orghr,ORGHR)
#define LAPACKsyevr_ SLEPC_BLASLAPACK(syevr,SYEVR)
#define LAPACKsygvd_ SLEPC_BLASLAPACK(sygvd,SYGVD)
#define LAPACKormlq_ SLEPC_BLASLAPACK(ormlq,ORMLQ)
#else
#define LAPACKorghr_ SLEPC_BLASLAPACK(unghr,UNGHR)
#define LAPACKsyevr_ SLEPC_BLASLAPACK(heevr,HEEVR)
#define LAPACKsygvd_ SLEPC_BLASLAPACK(hegvd,HEGVD)
#define LAPACKormlq_ SLEPC_BLASLAPACK(unmlq,UNMLQ)
#endif

#define LAPACKlamch_ SLEPC_BLASLAPACKREAL(lamch,LAMCH)
#define LAPACKstevr_ SLEPC_BLASLAPACKREAL(stevr,STEVR)

EXTERN_C_BEGIN

EXTERN PetscReal LAPACKlamch_(const char*,PetscBLASInt);
EXTERN PetscReal LAPACKlanhs_(const char*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscBLASInt);
EXTERN PetscReal LAPACKlange_(const char*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscBLASInt);
EXTERN void      LAPACKlaev2_(PetscScalar*,PetscScalar*,PetscScalar*,PetscReal*,PetscReal*,PetscReal*,PetscScalar*);
EXTERN void      LAPACKgehrd_(PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
EXTERN void      LAPACKorghr_(PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
EXTERN void      LAPACKgetri_(PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
EXTERN void      LAPACKstevr_(const char*,const char*,PetscBLASInt*,PetscReal*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscReal*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt,PetscBLASInt);
EXTERN void      LAPACKgelqf_(PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
EXTERN void      BLAStrsm_(const char*,const char*,const char*,const char*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt,PetscBLASInt,PetscBLASInt,PetscBLASInt);
EXTERN void      LAPACKormlq_(const char*,const char*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt,PetscBLASInt);

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
