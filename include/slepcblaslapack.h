/*

  Necessary routines in BLAS and LAPACK not included in petscblaslapack.f


   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2009, Universidad Politecnica de Valencia, Spain

   This file is part of SLEPc.
      
   SLEPc is free software: you can redistribute it and/or modify it under  the
   terms of version 3 of the GNU Lesser General Public License as published by
   the Free Software Foundation.

   SLEPc  is  distributed in the hope that it will be useful, but WITHOUT  ANY 
   WARRANTY;  without even the implied warranty of MERCHANTABILITY or  FITNESS 
   FOR  A  PARTICULAR PURPOSE. See the GNU Lesser General Public  License  for 
   more details.

   You  should have received a copy of the GNU Lesser General  Public  License
   along with SLEPc. If not, see <http://www.gnu.org/licenses/>.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

#if !defined(__SLEPCBLASLAPACK_H)
#define __SLEPCBLASLAPACK_H
#include "petscblaslapack.h"

/* Macros for building LAPACK names */
#if defined(PETSC_BLASLAPACK_UNDERSCORE)
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

#elif defined(PETSC_BLASLAPACK_CAPS) || defined(PETSC_BLASLAPACK_STDCALL)
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

/* LAPACK functions without string parameters */
#define LAPACKlaev2_ SLEPC_BLASLAPACK(laev2,LAEV2)
#define LAPACKgehrd_ SLEPC_BLASLAPACK(gehrd,GEHRD)
#define LAPACKgetri_ SLEPC_BLASLAPACK(getri,GETRI)
#define LAPACKgelqf_ SLEPC_BLASLAPACK(gelqf,GELQF)
#if !defined(PETSC_USE_COMPLEX)
#define LAPACKorghr_ SLEPC_BLASLAPACK(orghr,ORGHR)
#else
#define LAPACKorghr_ SLEPC_BLASLAPACK(unghr,UNGHR)
#endif
#define LAPACKtgexc_ SLEPC_BLASLAPACK(tgexc,TGEXC)
#define LAPACKlag2_  SLEPC_BLASLAPACKREAL(lag2,LAG2)

/* LAPACK functions with string parameters */
#if !defined(PETSC_BLASLAPACK_STDCALL)

#define BLAStrsm_(a,b,c,d,e,f,g,h,i,j,k) SLEPC_BLASLAPACK(trsm,TRSM) ((a),(b),(c),(d),(e),(f),(g),(h),(i),(j),(k),1,1,1,1)
#define LAPACKlanhs_(a,b,c,d,e) SLEPC_BLASLAPACK(lanhs,LANHS) ((a),(b),(c),(d),(e),1)
#define LAPACKlange_(a,b,c,d,e,f) SLEPC_BLASLAPACK(lange,LANGE) ((a),(b),(c),(d),(e),(f),1)
#define LAPACKstevr_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t) SLEPC_BLASLAPACKREAL(stevr,STEVR) ((a),(b),(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),(o),(p),(q),(r),(s),(t),1,1)
#define LAPACKbdsdc_(a,b,c,d,e,f,g,h,i,j,k,l,m,n) SLEPC_BLASLAPACKREAL(bdsdc,BDSDC) ((a),(b),(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),1,1)
#define LAPACKggevx_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z,aa,ab,ac) SLEPC_BLASLAPACK(ggevx,GGEVX) ((a),(b),(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),(o),(p),(q),(r),(s),(t),(u),(v),(w),(x),(y),(z),(aa),(ab),(ac),1,1,1,1)
#define LAPACKsteqr_(a,b,c,d,e,f,g,h) SLEPC_BLASLAPACKREAL(steqr,STEQR) ((a),(b),(c),(d),(e),(f),(g),(h),1)
#define LAPACKorgtr_(a,b,c,d,e,f,g,h) SLEPC_BLASLAPACKREAL(orgtr,ORGTR) ((a),(b),(c),(d),(e),(f),(g),(h),1)
#define LAPACKsytrd_(a,b,c,d,e,f,g,h,i,j) SLEPC_BLASLAPACKREAL(sytrd,SYTRD) ((a),(b),(c),(d),(e),(f),(g),(h),(i),(j),1)
#define LAPACKlamch_(a) SLEPC_BLASLAPACKREAL(lamch,LAMCH) ((a),1)

#if !defined(PETSC_USE_COMPLEX)
#define LAPACKsyevr_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u) SLEPC_BLASLAPACK(syevr,SYEVR) ((a),(b),(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),(o),(p),(q),(r),(s),(t),(u),1,1,1)
#define LAPACKsygvd_(a,b,c,d,e,f,g,h,i,j,k,l,m,n)  SLEPC_BLASLAPACK(sygvd,SYGVD) ((a),(b),(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),1,1)
#define LAPACKormlq_(a,b,c,d,e,f,g,h,i,j,k,l,m) SLEPC_BLASLAPACK(ormlq,ORMLQ) ((a),(b),(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),1,1)
#define LAPACKtrevc_(a,b,c,d,e,f,g,h,i,j,k,l,m,n) SLEPC_BLASLAPACK(trevc,TREVC) ((a),(b),(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),1,1)
#define LAPACKgeevx_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w) SLEPC_BLASLAPACK(geevx,GEEVX) ((a),(b),(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),(o),(p),(q),(r),(s),(t),(u),(v),(w),1,1,1,1)
#define LAPACKhseqr_(a,b,c,d,e,f,g,h,i,j,k,l,m,n) SLEPC_BLASLAPACK(hseqr,HSEQR) ((a),(b),(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),1,1)
#define LAPACKtrexc_(a,b,c,d,e,f,g,h,i,j) SLEPC_BLASLAPACK(trexc,TREXC) ((a),(b),(c),(d),(e),(f),(g),(h),(i),(j),1)
#define LAPACKgesdd_(a,b,c,d,e,f,g,h,i,j,k,l,m,n) SLEPC_BLASLAPACK(gesdd,GESDD) ((a),(b),(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),1)
#define LAPACKgges_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u) SLEPC_BLASLAPACK(gges,GGES) ((a),(b),(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),(o),(p),(q),(r),(s),(t),(u),1,1,1)
#else
#define LAPACKsyevr_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w) SLEPC_BLASLAPACK(heevr,HEEVR) ((a),(b),(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),(o),(p),(q),(r),(s),(t),(u),(v),(w),1,1,1)
#define LAPACKsygvd_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p) SLEPC_BLASLAPACK(hegvd,HEGVD) ((a),(b),(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),(o),(p),1,1)
#define LAPACKormlq_(a,b,c,d,e,f,g,h,i,j,k,l,m) SLEPC_BLASLAPACK(unmlq,UNMLQ) ((a),(b),(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),1,1)
#define LAPACKtrevc_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o) SLEPC_BLASLAPACK(trevc,TREVC) ((a),(b),(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),(o),1,1)
#define LAPACKgeevx_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v) SLEPC_BLASLAPACK(geevx,GEEVX) ((a),(b),(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),(o),(p),(q),(r),(s),(t),(u),(v),1,1,1,1)
#define LAPACKhseqr_(a,b,c,d,e,f,g,h,i,j,k,l,m) SLEPC_BLASLAPACK(hseqr,HSEQR) ((a),(b),(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),1,1)
#define LAPACKtrexc_(a,b,c,d,e,f,g,h,i) SLEPC_BLASLAPACK(trexc,TREXC) ((a),(b),(c),(d),(e),(f),(g),(h),(i),1)
#define LAPACKgesdd_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o) SLEPC_BLASLAPACK(gesdd,GESDD) ((a),(b),(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),(o),1)
#define LAPACKgges_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u) SLEPC_BLASLAPACK(gges,GGES) ((a),(b),(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),(o),(p),(q),(r),(s),(t),(u),1,1,1)
#endif

#else /* PETSC_BLASLAPACK_STDCALL */

#define BLAStrsm_(a,b,c,d,e,f,g,h,i,j,k) SLEPC_BLASLAPACK(trsm,TRSM) ((a),1,(b),1,(c),1,(d),1,(e),(f),(g),(h),(i),(j),(k))
#define LAPACKlanhs_(a,b,c,d,e) SLEPC_BLASLAPACK(lanhs,LANHS) ((a),1,(b),(c),(d),(e))
#define LAPACKlange_(a,b,c,d,e,f) SLEPC_BLASLAPACK(lange,LANGE) ((a),1,(b),(c),(d),(e),(f))
#define LAPACKstevr_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t) SLEPC_BLASLAPACKREAL(stevr,STEVR) ((a),1,(b),1,(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),(o),(p),(q),(r),(s),(t))
#define LAPACKbdsdc_(a,b,c,d,e,f,g,h,i,j,k,l,m,n) SLEPC_BLASLAPACKREAL(bdsdc,BDSDC) ((a),1,(b),1,(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n))
#define LAPACKggevx_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z,aa,ab,ac) SLEPC_BLASLAPACK(ggevx,GGEVX) ((a),1,(b),1,(c),1,(d),1,(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),(o),(p),(q),(r),(s),(t),(u),(v),(w),(x),(y),(z),(aa),(ab),(ac))
#define LAPACKsteqr_(a,b,c,d,e,f,g,h) SLEPC_BLASLAPACKREAL(steqr,STEQR) ((a),1,(b),(c),(d),(e),(f),(g),(h))
#define LAPACKorgtr_(a,b,c,d,e,f,g,h) SLEPC_BLASLAPACKREAL(orgtr,ORGTR) ((a),1,(b),(c),(d),(e),(f),(g),(h))
#define LAPACKsytrd_(a,b,c,d,e,f,g,h,i,j) SLEPC_BLASLAPACKREAL(sytrd,SYTRD) ((a),1,(b),(c),(d),(e),(f),(g),(h),(i),(j))
#define LAPACKlamch_(a) SLEPC_BLASLAPACKREAL(lamch,LAMCH) ((a),1)

#if !defined(PETSC_USE_COMPLEX)
#define LAPACKsyevr_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u) SLEPC_BLASLAPACK(syevr,SYEVR) ((a),1,(b),1,(c),1,(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),(o),(p),(q),(r),(s),(t),(u))
#define LAPACKsygvd_(a,b,c,d,e,f,g,h,i,j,k,l,m,n)  SLEPC_BLASLAPACK(sygvd,SYGVD) ((a),(b),1,(c),1,(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n))
#define LAPACKormlq_(a,b,c,d,e,f,g,h,i,j,k,l,m) SLEPC_BLASLAPACK(ormlq,ORMLQ) ((a),1,(b),1,(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m))
#define LAPACKtrevc_(a,b,c,d,e,f,g,h,i,j,k,l,m,n) SLEPC_BLASLAPACK(trevc,TREVC) ((a),1,(b),1,(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n))
#define LAPACKgeevx_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w) SLEPC_BLASLAPACK(geevx,GEEVX) ((a),1,(b),1,(c),1,(d),1,(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),(o),(p),(q),(r),(s),(t),(u),(v),(w))
#define LAPACKhseqr_(a,b,c,d,e,f,g,h,i,j,k,l,m,n) SLEPC_BLASLAPACK(hseqr,HSEQR) ((a),1,(b),1,(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n))
#define LAPACKtrexc_(a,b,c,d,e,f,g,h,i,j) SLEPC_BLASLAPACK(trexc,TREXC) ((a),1,(b),(c),(d),(e),(f),(g),(h),(i),(j))
#define LAPACKgesdd_(a,b,c,d,e,f,g,h,i,j,k,l,m,n) SLEPC_BLASLAPACK(gesdd,GESDD) ((a),1,(b),(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n))
#define LAPACKgges_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u) SLEPC_BLASLAPACK(gges,GGES) ((a),1,(b),1,(c),1,(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),(o),(p),(q),(r),(s),(t),(u))
#else
#define LAPACKsyevr_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w) SLEPC_BLASLAPACK(heevr,HEEVR) ((a),1,(b),1,(c),1,(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),(o),(p),(q),(r),(s),(t),(u),(v),(w))
#define LAPACKsygvd_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p) SLEPC_BLASLAPACK(hegvd,HEGVD) ((a),(b),1,(c),1,(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),(o),(p))
#define LAPACKormlq_(a,b,c,d,e,f,g,h,i,j,k,l,m) SLEPC_BLASLAPACK(unmlq,UNMLQ) ((a),1,(b),1,(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m))
#define LAPACKtrevc_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o) SLEPC_BLASLAPACK(trevc,TREVC) ((a),1,(b),1,(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),(o))
#define LAPACKgeevx_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v) SLEPC_BLASLAPACK(geevx,GEEVX) ((a),1,(b),1,(c),1,(d),1,(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),(o),(p),(q),(r),(s),(t),(u),(v))
#define LAPACKhseqr_(a,b,c,d,e,f,g,h,i,j,k,l,m) SLEPC_BLASLAPACK(hseqr,HSEQR) ((a),1,(b),1,(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m))
#define LAPACKtrexc_(a,b,c,d,e,f,g,h,i) SLEPC_BLASLAPACK(trexc,TREXC) ((a),1,(b),(c),(d),(e),(f),(g),(h),(i))
#define LAPACKgesdd_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o) SLEPC_BLASLAPACK(gesdd,GESDD) ((a),1,(b),(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),(o))
#define LAPACKgges_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u) SLEPC_BLASLAPACK(gges,GGES) ((a),1,(b),1,(c),1,(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),(o),(p),(q),(r),(s),(t),(u))
#endif

#endif

PETSC_EXTERN_CXX_BEGIN
EXTERN_C_BEGIN

#if !defined(PETSC_BLASLAPACK_STDCALL)

/* LAPACK functions without string parameters */
EXTERN void      SLEPC_BLASLAPACK(laev2,LAEV2) (PetscScalar*,PetscScalar*,PetscScalar*,PetscReal*,PetscReal*,PetscReal*,PetscScalar*);
EXTERN void      SLEPC_BLASLAPACK(gehrd,GEHRD) (PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
EXTERN void      SLEPC_BLASLAPACK(getri,GETRI) (PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
EXTERN void      SLEPC_BLASLAPACK(gelqf,GELQF) (PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
#if !defined(PETSC_USE_COMPLEX)
EXTERN void      SLEPC_BLASLAPACK(orghr,ORGHR) (PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
EXTERN void      SLEPC_BLASLAPACK(tgexc,TGEXC) (PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
#else
EXTERN void      SLEPC_BLASLAPACK(unghr,UNGHR) (PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
EXTERN void      SLEPC_BLASLAPACK(tgexc,TGEXC) (PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*);
#endif
EXTERN void      SLEPC_BLASLAPACKREAL(lag2,LAG2) (PetscReal*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscReal*,PetscReal*,PetscReal*,PetscReal*,PetscReal*,PetscReal*);

/* LAPACK functions with string parameters */
EXTERN void      SLEPC_BLASLAPACK(trsm,TRSM)   (const char*,const char*,const char*,const char*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt,PetscBLASInt,PetscBLASInt,PetscBLASInt);
EXTERN PetscReal SLEPC_BLASLAPACK(lanhs,LANHS) (const char*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscBLASInt);
EXTERN PetscReal SLEPC_BLASLAPACK(lange,LANGE) (const char*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscBLASInt);
EXTERN void      SLEPC_BLASLAPACKREAL(stevr,STEVR) (const char*,const char*,PetscBLASInt*,PetscReal*,PetscReal*,PetscReal*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscReal*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt,PetscBLASInt);
EXTERN void      SLEPC_BLASLAPACKREAL(bdsdc,BDSDC) (const char*,const char*,PetscBLASInt*,PetscReal*,PetscReal*,PetscReal*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt,PetscBLASInt);
EXTERN void      SLEPC_BLASLAPACKREAL(steqr,STEQR) (const char*,PetscBLASInt*,PetscReal*,PetscReal*,PetscReal*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscBLASInt);
EXTERN void      SLEPC_BLASLAPACKREAL(orgtr,ORGTR) (const char*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscReal*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt);
EXTERN void      SLEPC_BLASLAPACKREAL(sytrd,SYTRD) (const char*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscReal*,PetscReal*,PetscReal*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt);
EXTERN PetscReal SLEPC_BLASLAPACKREAL(lamch,LAMCH) (const char*,PetscBLASInt);

#if !defined(PETSC_USE_COMPLEX)
EXTERN void      SLEPC_BLASLAPACK(hseqr,HSEQR) (const char*,const char*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt,PetscBLASInt);
EXTERN void      SLEPC_BLASLAPACK(trexc,TREXC) (const char*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt);
EXTERN void      SLEPC_BLASLAPACK(trevc,TREVC) (const char*,const char*,PetscTruth*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt,PetscBLASInt);
EXTERN void      SLEPC_BLASLAPACK(geevx,GEEVX) (const char*,const char*,const char*,const char*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt,PetscBLASInt,PetscBLASInt,PetscBLASInt);
EXTERN void      SLEPC_BLASLAPACK(ggevx,GGEVX) (const char*,const char*,const char*,const char*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt,PetscBLASInt,PetscBLASInt,PetscBLASInt);
EXTERN void      SLEPC_BLASLAPACK(syevr,SYEVR) (const char*,const char*,const char*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt,PetscBLASInt,PetscBLASInt);        
EXTERN void      SLEPC_BLASLAPACK(sygvd,SYGVD) (PetscBLASInt*,const char*,const char*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt,PetscBLASInt);
EXTERN void      SLEPC_BLASLAPACK(gesdd,GESDD) (const char*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt);
EXTERN void      SLEPC_BLASLAPACK(ormlq,ORMLQ) (const char*,const char*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt,PetscBLASInt);
EXTERN void      SLEPC_BLASLAPACK(gges,GGES) (const char*,const char*,const char*,PetscBLASInt(*)(),PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt,PetscBLASInt,PetscBLASInt);
#else
EXTERN void      SLEPC_BLASLAPACK(hseqr,HSEQR) (const char*,const char*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt,PetscBLASInt);
EXTERN void      SLEPC_BLASLAPACK(trexc,TREXC) (const char*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt);
EXTERN void      SLEPC_BLASLAPACK(trevc,TREVC) (const char*,const char*,PetscTruth*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscReal*,PetscBLASInt*,PetscBLASInt,PetscBLASInt);
EXTERN void      SLEPC_BLASLAPACK(geevx,GEEVX) (const char*,const char*,const char*,const char*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscReal*,PetscReal*,PetscReal*,PetscReal*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscBLASInt,PetscBLASInt,PetscBLASInt,PetscBLASInt);
EXTERN void      SLEPC_BLASLAPACK(ggevx,GGEVX) (const char*,const char*,const char*,const char*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*, PetscBLASInt*,PetscReal*,PetscReal*,PetscReal*,PetscReal*,PetscReal*,PetscReal*,PetscScalar*, PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt,PetscBLASInt,PetscBLASInt,PetscBLASInt);
EXTERN void      SLEPC_BLASLAPACK(heevr,HEEVR) (const char *,const char*,const char*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscReal*,PetscScalar*,PetscBLASInt*,PetscBLASInt*, PetscScalar*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt,PetscBLASInt,PetscBLASInt);
EXTERN void      SLEPC_BLASLAPACK(hegvd,HEGVD) (PetscBLASInt*,const char*,const char*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt,PetscBLASInt);
EXTERN void      SLEPC_BLASLAPACK(gesdd,GESDD) (const char*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt);
EXTERN void      SLEPC_BLASLAPACK(unmlq,UNMLQ) (const char*,const char*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt,PetscBLASInt);
EXTERN void      SLEPC_BLASLAPACK(ungtr,UNGTR) (const char*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt);
EXTERN void      SLEPC_BLASLAPACK(hetrd,HETRD) (const char*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscReal*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt);
EXTERN void      SLEPC_BLASLAPACK(gges,GGES) (const char*,const char*,const char*,PetscBLASInt(*)(),PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt,PetscBLASInt,PetscBLASInt);        
#endif

#else /* PETSC_BLASLAPACK_STDCALL */

/* LAPACK functions without string parameters */
EXTERN void PETSC_STDCALL SLEPC_BLASLAPACK(laev2,LAEV2) (PetscScalar*,PetscScalar*,PetscScalar*,PetscReal*,PetscReal*,PetscReal*,PetscScalar*);
EXTERN void PETSC_STDCALL SLEPC_BLASLAPACK(gehrd,GEHRD) (PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
EXTERN void PETSC_STDCALL SLEPC_BLASLAPACK(getri,GETRI) (PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
EXTERN void PETSC_STDCALL SLEPC_BLASLAPACK(gelqf,GELQF) (PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
#if !defined(PETSC_USE_COMPLEX)
EXTERN void PETSC_STDCALL SLEPC_BLASLAPACK(orghr,ORGHR) (PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
EXTERN void PETSC_STDCALL SLEPC_BLASLAPACK(tgexc,TGEXC) (PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
#else
EXTERN void PETSC_STDCALL SLEPC_BLASLAPACK(unghr,UNGHR) (PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
EXTERN void PETSC_STDCALL SLEPC_BLASLAPACK(tgexc,TGEXC) (PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*);
#endif
EXTERN void PETSC_STDCALL SLEPC_BLASLAPACKREAL(lag2,LAG2) (PetscReal*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscReal*,PetscReal*,PetscReal*,PetscReal*,PetscReal*,PetscReal*);

/* LAPACK functions with string parameters */
EXTERN void PETSC_STDCALL SLEPC_BLASLAPACK(trsm,TRSM)   (const char*,PetscBLASInt,const char*,PetscBLASInt,const char*,PetscBLASInt,const char*,PetscBLASInt,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*);
EXTERN PetscReal PETSC_STDCALL SLEPC_BLASLAPACK(lanhs,LANHS) (const char*,PetscBLASInt,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*);
EXTERN PetscReal PETSC_STDCALL SLEPC_BLASLAPACK(lange,LANGE) (const char*,PetscBLASInt,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*);
EXTERN void PETSC_STDCALL SLEPC_BLASLAPACKREAL(stevr,STEVR) (const char*,PetscBLASInt,const char*,PetscBLASInt,PetscBLASInt*,PetscReal*,PetscReal*,PetscReal*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscReal*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*);
EXTERN void PETSC_STDCALL SLEPC_BLASLAPACKREAL(bdsdc,BDSDC) (const char*,PetscBLASInt,const char*,PetscBLASInt,PetscBLASInt*,PetscReal*,PetscReal*,PetscReal*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscBLASInt*);
EXTERN void PETSC_STDCALL SLEPC_BLASLAPACKREAL(steqr,STEQR) (const char*,PetscBLASInt,PetscBLASInt*,PetscReal*,PetscReal*,PetscReal*,PetscBLASInt*,PetscReal*,PetscBLASInt*);
EXTERN void PETSC_STDCALL SLEPC_BLASLAPACKREAL(orgtr,ORGTR) (const char*,PetscBLASInt,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscReal*,PetscReal*,PetscBLASInt*,PetscBLASInt*);
EXTERN void PETSC_STDCALL SLEPC_BLASLAPACKREAL(sytrd,SYTRD) (const char*,PetscBLASInt,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscReal*,PetscReal*,PetscReal*,PetscReal*,PetscBLASInt*,PetscBLASInt*);
EXTERN PetscReal SLEPC_BLASLAPACKREAL(lamch,LAMCH) (const char*,PetscBLASInt);

#if !defined(PETSC_USE_COMPLEX)
EXTERN void PETSC_STDCALL SLEPC_BLASLAPACK(hseqr,HSEQR) (const char*,PetscBLASInt,const char*,PetscBLASInt,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
EXTERN void PETSC_STDCALL SLEPC_BLASLAPACK(trexc,TREXC) (const char*,PetscBLASInt,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*);
EXTERN void PETSC_STDCALL SLEPC_BLASLAPACK(trevc,TREVC) (const char*,PetscBLASInt,const char*,PetscBLASInt,PetscTruth*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*);
EXTERN void PETSC_STDCALL SLEPC_BLASLAPACK(geevx,GEEVX) (const char*,PetscBLASInt,const char*,PetscBLASInt,const char*,PetscBLASInt,const char*,PetscBLASInt,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*);
EXTERN void PETSC_STDCALL SLEPC_BLASLAPACK(ggevx,GGEVX) (const char*,PetscBLASInt,const char*,PetscBLASInt,const char*,PetscBLASInt,const char*,PetscBLASInt,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*);
EXTERN void PETSC_STDCALL SLEPC_BLASLAPACK(syevr,SYEVR) (const char*,PetscBLASInt,const char*,PetscBLASInt,const char*,PetscBLASInt,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*);
EXTERN void PETSC_STDCALL SLEPC_BLASLAPACK(sygvd,SYGVD) (PetscBLASInt*,const char*,PetscBLASInt,const char*,PetscBLASInt,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*);
EXTERN void PETSC_STDCALL SLEPC_BLASLAPACK(gesdd,GESDD) (const char*,PetscBLASInt,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*);
EXTERN void PETSC_STDCALL SLEPC_BLASLAPACK(ormlq,ORMLQ) (const char*,PetscBLASInt,const char*,PetscBLASInt,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
EXTERN void PETSC_STDCALL SLEPC_BLASLAPACK(gges,GGES) (const char*,PetscBLASInt,const char*,PetscBLASInt,const char*,PetscBLASInt,PetscBLASInt(*)(...),PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*);
#else
EXTERN void PETSC_STDCALL SLEPC_BLASLAPACK(hseqr,HSEQR) (const char*,PetscBLASInt,const char*,PetscBLASInt,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
EXTERN void PETSC_STDCALL SLEPC_BLASLAPACK(trexc,TREXC) (const char*,PetscBLASInt,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*);
EXTERN void PETSC_STDCALL SLEPC_BLASLAPACK(trevc,TREVC) (const char*,PetscBLASInt,const char*,PetscBLASInt,PetscTruth*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscReal*,PetscBLASInt*);
EXTERN void PETSC_STDCALL SLEPC_BLASLAPACK(geevx,GEEVX) (const char*,PetscBLASInt,const char*,PetscBLASInt,const char*,PetscBLASInt,const char*,PetscBLASInt,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscReal*,PetscReal*,PetscReal*,PetscReal*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscBLASInt*);
EXTERN void PETSC_STDCALL SLEPC_BLASLAPACK(ggevx,GGEVX) (const char*,PetscBLASInt,const char*,PetscBLASInt,const char*,PetscBLASInt,const char*,PetscBLASInt,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*, PetscBLASInt*,PetscReal*,PetscReal*,PetscReal*,PetscReal*,PetscReal*,PetscReal*,PetscScalar*, PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*);
EXTERN void PETSC_STDCALL SLEPC_BLASLAPACK(heevr,HEEVR) (const char*,PetscBLASInt,const char*,PetscBLASInt,const char*,PetscBLASInt,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscReal*,PetscScalar*,PetscBLASInt*,PetscBLASInt*, PetscScalar*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*);
EXTERN void PETSC_STDCALL SLEPC_BLASLAPACK(hegvd,HEGVD) (PetscBLASInt*,const char*,PetscBLASInt,const char*,PetscBLASInt,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*);
EXTERN void PETSC_STDCALL SLEPC_BLASLAPACK(gesdd,GESDD) (const char*,PetscBLASInt,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscBLASInt*);
EXTERN void PETSC_STDCALL SLEPC_BLASLAPACK(unmlq,UNMLQ) (const char*,PetscBLASInt,const char*,PetscBLASInt,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
EXTERN void PETSC_STDCALL SLEPC_BLASLAPACK(ungtr,UNGTR) (const char*,PetscBLASInt,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
EXTERN void PETSC_STDCALL SLEPC_BLASLAPACK(hetrd,HETRD) (const char*,PetscBLASInt,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscReal*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
EXTERN void PETSC_STDCALL SLEPC_BLASLAPACK(gges,GGES) (const char*,PetscBLASInt,const char*,PetscBLASInt,const char*,PetscBLASInt,PetscBLASInt(*)(...),PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscBLASInt*);
#endif

#endif

EXTERN_C_END
PETSC_EXTERN_CXX_END

#endif
