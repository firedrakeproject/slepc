/*
   Necessary routines in BLAS and LAPACK not included in petscblaslapack.f

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2016, Universitat Politecnica de Valencia, Spain

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
#include <petscblaslapack.h>

/* Macros for building LAPACK names */
#if defined(PETSC_BLASLAPACK_UNDERSCORE)
#if defined(PETSC_USE_REAL_SINGLE)
#define SLEPC_BLASLAPACKREAL(lcase,ucase) s##lcase##_
#if defined(PETSC_USE_COMPLEX)
#define SLEPC_BLASLAPACK(lcase,ucase) c##lcase##_
#else
#define SLEPC_BLASLAPACK(lcase,ucase) s##lcase##_
#endif
#elif defined(PETSC_USE_REAL___FLOAT128)
#define SLEPC_BLASLAPACKREAL(lcase,ucase) q##lcase##_
#if defined(PETSC_USE_COMPLEX)
#define SLEPC_BLASLAPACK(lcase,ucase) w##lcase##_
#else
#define SLEPC_BLASLAPACK(lcase,ucase) q##lcase##_
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
#if defined(PETSC_USE_REAL_SINGLE)
#define SLEPC_BLASLAPACKREAL(lcase,ucase) S##ucase
#if defined(PETSC_USE_COMPLEX)
#define SLEPC_BLASLAPACK(lcase,ucase) C##ucase
#else
#define SLEPC_BLASLAPACK(lcase,ucase) S##ucase
#endif
#elif defined(PETSC_USE_REAL___FLOAT128)
#define SLEPC_BLASLAPACKREAL(lcase,ucase) Q##ucase
#if defined(PETSC_USE_COMPLEX)
#define SLEPC_BLASLAPACK(lcase,ucase) W##ucase
#else
#define SLEPC_BLASLAPACK(lcase,ucase) Q##ucase
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
#if defined(PETSC_USE_REAL_SINGLE)
#define SLEPC_BLASLAPACKREAL(lcase,ucase) s##lcase
#if defined(PETSC_USE_COMPLEX)
#define SLEPC_BLASLAPACK(lcase,ucase) c##lcase
#else
#define SLEPC_BLASLAPACK(lcase,ucase) s##lcase
#endif
#elif defined(PETSC_USE_REAL___FLOAT128)
#define SLEPC_BLASLAPACKREAL(lcase,ucase) q##lcase
#if defined(PETSC_USE_COMPLEX)
#define SLEPC_BLASLAPACK(lcase,ucase) w##lcase
#else
#define SLEPC_BLASLAPACK(lcase,ucase) q##lcase
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

/* LAPACK return type: we assume slange, etc. behave in the same way as snrm2 */
#if defined(PETSC_USE_REAL_SINGLE) && defined(PETSC_BLASLAPACK_SNRM2_RETURNS_DOUBLE)
#define SlepcLRT double
#else
#define SlepcLRT PetscReal
#endif

/* LAPACK functions without string parameters */
#define LAPACKlaev2_ SLEPC_BLASLAPACK(laev2,LAEV2)
#define LAPACKgehrd_ SLEPC_BLASLAPACK(gehrd,GEHRD)
#define LAPACKgelqf_ SLEPC_BLASLAPACK(gelqf,GELQF)
#define LAPACKgeqp3_ SLEPC_BLASLAPACK(geqp3,GEQP3)
#define LAPACKtgexc_ SLEPC_BLASLAPACK(tgexc,TGEXC)
#define LAPACKlarfg_ SLEPC_BLASLAPACK(larfg,LARFG)
#define LAPACKlag2_  SLEPC_BLASLAPACKREAL(lag2,LAG2)
#define LAPACKlasv2_ SLEPC_BLASLAPACKREAL(lasv2,LASV2)
#define LAPACKlartg_ SLEPC_BLASLAPACKREAL(lartg,LARTG)
#define LAPACKlaln2_ SLEPC_BLASLAPACKREAL(laln2,LALN2)
#define LAPACKlaed4_ SLEPC_BLASLAPACKREAL(laed4,LAED4)
#define LAPACKlamrg_ SLEPC_BLASLAPACKREAL(lamrg,LAMRG)
#define LAPACKlapy2_ SLEPC_BLASLAPACKREAL(lapy2,LAPY2)
#if !defined(PETSC_USE_COMPLEX)
#define LAPACKorghr_ SLEPC_BLASLAPACK(orghr,ORGHR)
#else
#define LAPACKorghr_ SLEPC_BLASLAPACK(unghr,UNGHR)
#endif
/* the next one needs a special treatment due to the special names:
   srot, drot, csrot, zdrot */
#if !defined(PETSC_USE_COMPLEX)
#define BLASrot_     SLEPC_BLASLAPACK(rot,ROT)
#else
#if defined(PETSC_USE_REAL_SINGLE)
#define BLASrot_     SLEPC_BLASLAPACK(srot,SROT)
#elif defined(PETSC_USE_REAL___FLOAT128)
#define BLASrot_     SLEPC_BLASLAPACK(qrot,QROT)
#else
#define BLASrot_     SLEPC_BLASLAPACK(drot,DROT)
#endif
#endif

/* LAPACK functions with string parameters */
#if !defined(PETSC_BLASLAPACK_STDCALL)

/* same name for real and complex */
#define LAPACKlanhs_(a,b,c,d,e) SLEPC_BLASLAPACK(lanhs,LANHS) ((a),(b),(c),(d),(e),1)
#define LAPACKlange_(a,b,c,d,e,f) SLEPC_BLASLAPACK(lange,LANGE) ((a),(b),(c),(d),(e),(f),1)
#define LAPACKggevx_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z,aa,ab,ac) SLEPC_BLASLAPACK(ggevx,GGEVX) ((a),(b),(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),(o),(p),(q),(r),(s),(t),(u),(v),(w),(x),(y),(z),(aa),(ab),(ac),1,1,1,1)
#define LAPACKggev_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q) SLEPC_BLASLAPACK(ggev,GGEV) ((a),(b),(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),(o),(p),(q),1,1)
#define LAPACKpbtrf_(a,b,c,d,e,f) SLEPC_BLASLAPACK(pbtrf,PBTRF) ((a),(b),(c),(d),(e),(f),1)
#define LAPACKlarf_(a,b,c,d,e,f,g,h,i) SLEPC_BLASLAPACK(larf,LARF) ((a),(b),(c),(d),(e),(f),(g),(h),(i),1)
#define BLAStrmm_(a,b,c,d,e,f,g,h,i,j,k) SLEPC_BLASLAPACK(trmm,TRMM) ((a),(b),(c),(d),(e),(f),(g),(h),(i),(j),(k),1,1,1,1)
#define LAPACKlacpy_(a,b,c,d,e,f,g) SLEPC_BLASLAPACK(lacpy,LACPY) ((a),(b),(c),(d),(e),(f),(g),1)
#define LAPACKlascl_(a,b,c,d,e,f,g,h,i,j) SLEPC_BLASLAPACK(lascl,LASCL) ((a),(b),(c),(d),(e),(f),(g),(h),(i),(j),1)
#define LAPACKlansy_(a,b,c,d,e,f) SLEPC_BLASLAPACK(lansy,LANSY) ((a),(b),(c),(d),(e),(f),1,1)
#define LAPACKlaset_(a,b,c,d,e,f,g) SLEPC_BLASLAPACK(laset,LASET) ((a),(b),(c),(d),(e),(f),(g),1)
#define LAPACKtrsyl_(a,b,c,d,e,f,g,h,i,j,k,l,m) SLEPC_BLASLAPACK(trsyl,TRSYL) ((a),(b),(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),1,1)
#define LAPACKtrtri_(a,b,c,d,e,f) SLEPC_BLASLAPACK(trtri,TRTRI) ((a),(b),(c),(d),(e),(f),1,1)
/* subroutines in which we use only the real version, do not care whether they have different name */
#define LAPACKstevr_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t) SLEPC_BLASLAPACKREAL(stevr,STEVR) ((a),(b),(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),(o),(p),(q),(r),(s),(t),1,1)
#define LAPACKbdsdc_(a,b,c,d,e,f,g,h,i,j,k,l,m,n) SLEPC_BLASLAPACKREAL(bdsdc,BDSDC) ((a),(b),(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),1,1)
#define LAPACKlamch_(a) SLEPC_BLASLAPACKREAL(lamch,LAMCH) ((a),1)
#define LAPACKlamc3_(a,b) SLEPC_BLASLAPACKREAL(lamc3,LAMC3) ((a),(b))

#if !defined(PETSC_USE_COMPLEX)
/* different name or signature, real */
#define LAPACKsyevr_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u) SLEPC_BLASLAPACK(syevr,SYEVR) ((a),(b),(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),(o),(p),(q),(r),(s),(t),(u),1,1,1)
#define LAPACKsyevd_(a,b,c,d,e,f,g,h,i,j,k) SLEPC_BLASLAPACK(syevd,SYEVD) ((a),(b),(c),(d),(e),(f),(g),(h),(i),(j),(k),1,1)
#define LAPACKsygvd_(a,b,c,d,e,f,g,h,i,j,k,l,m,n)  SLEPC_BLASLAPACK(sygvd,SYGVD) ((a),(b),(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),1,1)
#define LAPACKormlq_(a,b,c,d,e,f,g,h,i,j,k,l,m) SLEPC_BLASLAPACK(ormlq,ORMLQ) ((a),(b),(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),1,1)
#define LAPACKorgtr_(a,b,c,d,e,f,g,h) SLEPC_BLASLAPACK(orgtr,ORGTR) ((a),(b),(c),(d),(e),(f),(g),(h),1)
#define LAPACKsytrd_(a,b,c,d,e,f,g,h,i,j) SLEPC_BLASLAPACK(sytrd,SYTRD) ((a),(b),(c),(d),(e),(f),(g),(h),(i),(j),1)
#define LAPACKtrevc_(a,b,c,d,e,f,g,h,i,j,k,l,m,n) SLEPC_BLASLAPACK(trevc,TREVC) ((a),(b),(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),1,1)
#define LAPACKgeevx_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w) SLEPC_BLASLAPACK(geevx,GEEVX) ((a),(b),(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),(o),(p),(q),(r),(s),(t),(u),(v),(w),1,1,1,1)
#define LAPACKgees_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o) SLEPC_BLASLAPACK(gees,GEES) ((a),(b),(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),(o),1,1)
#define LAPACKtrexc_(a,b,c,d,e,f,g,h,i,j) SLEPC_BLASLAPACK(trexc,TREXC) ((a),(b),(c),(d),(e),(f),(g),(h),(i),(j),1)
#define LAPACKgesdd_(a,b,c,d,e,f,g,h,i,j,k,l,m,n) SLEPC_BLASLAPACK(gesdd,GESDD) ((a),(b),(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),1)
#define LAPACKtgevc_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p) SLEPC_BLASLAPACK(tgevc,TGEVC) ((a),(b),(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),(o),(p),1,1)
#define LAPACKhsein_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s) SLEPC_BLASLAPACK(hsein,HSEIN) ((a),(b),(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),(o),(p),(q),(r),(s),1,1,1)
#define LAPACKstedc_(a,b,c,d,e,f,g,h,i,j,k) SLEPC_BLASLAPACK(stedc,STEDC) ((a),(b),(c),(d),(e),(f),(g),(h),(i),(j),(k),1)
#else
/* different name or signature, complex */
#define LAPACKsyevr_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w) SLEPC_BLASLAPACK(heevr,HEEVR) ((a),(b),(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),(o),(p),(q),(r),(s),(t),(u),(v),(w),1,1,1)
#define LAPACKsyevd_(a,b,c,d,e,f,g,h,i,j,k,l,m) SLEPC_BLASLAPACK(heevd,HEEVD) ((a),(b),(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),1,1)
#define LAPACKsygvd_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p) SLEPC_BLASLAPACK(hegvd,HEGVD) ((a),(b),(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),(o),(p),1,1)
#define LAPACKormlq_(a,b,c,d,e,f,g,h,i,j,k,l,m) SLEPC_BLASLAPACK(unmlq,UNMLQ) ((a),(b),(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),1,1)
#define LAPACKorgtr_(a,b,c,d,e,f,g,h) SLEPC_BLASLAPACK(ungtr,UNGTR) ((a),(b),(c),(d),(e),(f),(g),(h),1)
#define LAPACKsytrd_(a,b,c,d,e,f,g,h,i,j) SLEPC_BLASLAPACK(hetrd,HETRD) ((a),(b),(c),(d),(e),(f),(g),(h),(i),(j),1)
#define LAPACKtrevc_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o) SLEPC_BLASLAPACK(trevc,TREVC) ((a),(b),(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),(o),1,1)
#define LAPACKgeevx_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v) SLEPC_BLASLAPACK(geevx,GEEVX) ((a),(b),(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),(o),(p),(q),(r),(s),(t),(u),(v),1,1,1,1)
#define LAPACKgees_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o) SLEPC_BLASLAPACK(gees,GEES) ((a),(b),(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),(o),1,1)
#define LAPACKtrexc_(a,b,c,d,e,f,g,h,i) SLEPC_BLASLAPACK(trexc,TREXC) ((a),(b),(c),(d),(e),(f),(g),(h),(i),1)
#define LAPACKgesdd_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o) SLEPC_BLASLAPACK(gesdd,GESDD) ((a),(b),(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),(o),1)
#define LAPACKtgevc_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q) SLEPC_BLASLAPACK(tgevc,TGEVC) ((a),(b),(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),(o),(p),(q),1,1)
#define LAPACKhsein_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s) SLEPC_BLASLAPACK(hsein,HSEIN) ((a),(b),(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),(o),(p),(q),(r),(s),1,1,1)
#define LAPACKstedc_(a,b,c,d,e,f,g,h,i,j,k,l,m) SLEPC_BLASLAPACK(stedc,STEDC) ((a),(b),(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),1)
#endif

#else /* PETSC_BLASLAPACK_STDCALL */

/* same name for real and complex */
#define LAPACKlanhs_(a,b,c,d,e) SLEPC_BLASLAPACK(lanhs,LANHS) ((a),1,(b),(c),(d),(e))
#define LAPACKlange_(a,b,c,d,e,f) SLEPC_BLASLAPACK(lange,LANGE) ((a),1,(b),(c),(d),(e),(f))
#define LAPACKggevx_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z,aa,ab,ac) SLEPC_BLASLAPACK(ggevx,GGEVX) ((a),1,(b),1,(c),1,(d),1,(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),(o),(p),(q),(r),(s),(t),(u),(v),(w),(x),(y),(z),(aa),(ab),(ac))
#define LAPACKggev_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q) SLEPC_BLASLAPACK(ggev,GGEV) ((a),1,(b),1,(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),(o),(p),(q))
#define LAPACKpbtrf_(a,b,c,d,e,f) SLEPC_BLASLAPACK(pbtrf,PBTRF) ((a),1,(b),(c),(d),(e),(f))
#define LAPACKlarf_(a,b,c,d,e,f,g,h,i) SLEPC_BLASLAPACK(larf,LARF) ((a),1,(b),(c),(d),(e),(f),(g),(h),(i))
#define BLAStrmm_(a,b,c,d,e,f,g,h,i,j,k) SLEPC_BLASLAPACK(trmm,TRMM) ((a),1,(b),1,(c),1,(d),1,(e),(f),(g),(h),(i),(j),(k))
#define LAPACKlacpy_(a,b,c,d,e,f,g) SLEPC_BLASLAPACK(lacpy,LACPY) ((a),1,(b),(c),(d),(e),(f),(g))
#define LAPACKlascl_(a,b,c,d,e,f,g,h,i,j) SLEPC_BLASLAPACK(lascl,LASCL) ((a),1,(b),(c),(d),(e),(f),(g),(h),(i),(j))
#define LAPACKlansy_(a,b,c,d,e,f) SLEPC_BLASLAPACK(lansy,LANSY) ((a),1,(b),1,(c),(d),(e),(f))
#define LAPACKlaset_(a,b,c,d,e,f,g) SLEPC_BLASLAPACK(laset,LASET) ((a),1,(b),(c),(d),(e),(f),(g))
#define LAPACKtrsyl_(a,b,c,d,e,f,g,h,i,j,k,l,m) SLEPC_BLASLAPACK(trsyl,TRSYL) ((a),1,(b),1,(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m))
#define LAPACKtrtri_(a,b,c,d,e,f) SLEPC_BLASLAPACK(trtri,TRTRI) ((a),1,(b),1,(c),(d),(e),(f))
/* subroutines in which we use only the real version, do not care whether they have different name */
#define LAPACKstevr_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t) SLEPC_BLASLAPACKREAL(stevr,STEVR) ((a),1,(b),1,(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),(o),(p),(q),(r),(s),(t))
#define LAPACKbdsdc_(a,b,c,d,e,f,g,h,i,j,k,l,m,n) SLEPC_BLASLAPACKREAL(bdsdc,BDSDC) ((a),1,(b),1,(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n))
#define LAPACKlamch_(a) SLEPC_BLASLAPACKREAL(lamch,LAMCH) ((a),1)
#define LAPACKlamc3_(a,b) SLEPC_BLASLAPACKREAL(lamc3,LAMC3) ((a),(b))

#if !defined(PETSC_USE_COMPLEX)
/* different name or signature, real */
#define LAPACKsyevr_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u) SLEPC_BLASLAPACK(syevr,SYEVR) ((a),1,(b),1,(c),1,(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),(o),(p),(q),(r),(s),(t),(u))
#define LAPACKsyevd_(a,b,c,d,e,f,g,h,i,j,k) SLEPC_BLASLAPACK(syevd,SYEVD) ((a),1,(b),1,(c),(d),(e),(f),(g),(h),(i),(j),(k))
#define LAPACKsygvd_(a,b,c,d,e,f,g,h,i,j,k,l,m,n)  SLEPC_BLASLAPACK(sygvd,SYGVD) ((a),(b),1,(c),1,(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n))
#define LAPACKormlq_(a,b,c,d,e,f,g,h,i,j,k,l,m) SLEPC_BLASLAPACK(ormlq,ORMLQ) ((a),1,(b),1,(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m))
#define LAPACKorgtr_(a,b,c,d,e,f,g,h) SLEPC_BLASLAPACK(orgtr,ORGTR) ((a),1,(b),(c),(d),(e),(f),(g),(h))
#define LAPACKsytrd_(a,b,c,d,e,f,g,h,i,j) SLEPC_BLASLAPACK(sytrd,SYTRD) ((a),1,(b),(c),(d),(e),(f),(g),(h),(i),(j))
#define LAPACKtrevc_(a,b,c,d,e,f,g,h,i,j,k,l,m,n) SLEPC_BLASLAPACK(trevc,TREVC) ((a),1,(b),1,(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n))
#define LAPACKgeevx_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w) SLEPC_BLASLAPACK(geevx,GEEVX) ((a),1,(b),1,(c),1,(d),1,(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),(o),(p),(q),(r),(s),(t),(u),(v),(w))
#define LAPACKgees_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o) SLEPC_BLASLAPACK(gees,GEES) ((a),1,(b),1,(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),(o))
#define LAPACKtrexc_(a,b,c,d,e,f,g,h,i,j) SLEPC_BLASLAPACK(trexc,TREXC) ((a),1,(b),(c),(d),(e),(f),(g),(h),(i),(j))
#define LAPACKgesdd_(a,b,c,d,e,f,g,h,i,j,k,l,m,n) SLEPC_BLASLAPACK(gesdd,GESDD) ((a),1,(b),(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n))
#define LAPACKtgevc_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p) SLEPC_BLASLAPACK(tgevc,TGEVC) ((a),1,(b),1,(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),(o),(p))
#define LAPACKhsein_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s) SLEPC_BLASLAPACK(hsein,HSEIN) ((a),1,(b),1,(c),1,(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),(o),(p),(q),(r),(s))
#define LAPACKstedc_(a,b,c,d,e,f,g,h,i,j,k) SLEPC_BLASLAPACK(stedc,STEDC) ((a),1,(b),(c),(d),(e),(f),(g),(h),(i),(j),(k))
#else
/* different name or signature, complex */
#define LAPACKsyevr_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w) SLEPC_BLASLAPACK(heevr,HEEVR) ((a),1,(b),1,(c),1,(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),(o),(p),(q),(r),(s),(t),(u),(v),(w))
#define LAPACKsyevd_(a,b,c,d,e,f,g,h,i,j,k,l,m) SLEPC_BLASLAPACK(heevd,HEEVD) ((a),1,(b),1,(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m))
#define LAPACKsygvd_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p) SLEPC_BLASLAPACK(hegvd,HEGVD) ((a),(b),1,(c),1,(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),(o),(p))
#define LAPACKormlq_(a,b,c,d,e,f,g,h,i,j,k,l,m) SLEPC_BLASLAPACK(unmlq,UNMLQ) ((a),1,(b),1,(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m))
#define LAPACKorgtr_(a,b,c,d,e,f,g,h) SLEPC_BLASLAPACK(ungtr,UNGTR) ((a),1,(b),(c),(d),(e),(f),(g),(h))
#define LAPACKsytrd_(a,b,c,d,e,f,g,h,i,j) SLEPC_BLASLAPACK(hetrd,HETRD) ((a),1,(b),(c),(d),(e),(f),(g),(h),(i),(j))
#define LAPACKtrevc_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o) SLEPC_BLASLAPACK(trevc,TREVC) ((a),1,(b),1,(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),(o))
#define LAPACKgeevx_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v) SLEPC_BLASLAPACK(geevx,GEEVX) ((a),1,(b),1,(c),1,(d),1,(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),(o),(p),(q),(r),(s),(t),(u),(v))
#define LAPACKgees_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o) SLEPC_BLASLAPACK(gees,GEES) ((a),1,(b),1,(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),(o))
#define LAPACKtrexc_(a,b,c,d,e,f,g,h,i) SLEPC_BLASLAPACK(trexc,TREXC) ((a),1,(b),(c),(d),(e),(f),(g),(h),(i))
#define LAPACKgesdd_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o) SLEPC_BLASLAPACK(gesdd,GESDD) ((a),1,(b),(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),(o))
#define LAPACKtgevc_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q) SLEPC_BLASLAPACK(tgevc,TGEVC) ((a),1,(b),1,(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),(o),(p),(q))
#define LAPACKhsein_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s) SLEPC_BLASLAPACK(hsein,HSEIN) ((a),1,(b),1,(c),1,(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),(o),(p),(q),(r),(s))
#define LAPACKstedc_(a,b,c,d,e,f,g,h,i,j,k,l,m) SLEPC_BLASLAPACK(stedc,STEDC) ((a),1,(b),(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m))
#endif

#endif

#if !defined(PETSC_BLASLAPACK_STDCALL)

/* LAPACK functions without string parameters */
PETSC_EXTERN void      SLEPC_BLASLAPACK(laev2,LAEV2) (PetscScalar*,PetscScalar*,PetscScalar*,PetscReal*,PetscReal*,PetscReal*,PetscScalar*);
PETSC_EXTERN void      SLEPC_BLASLAPACK(gehrd,GEHRD) (PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN void      SLEPC_BLASLAPACK(gelqf,GELQF) (PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN void      SLEPC_BLASLAPACK(larfg,LARFG) (PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*);
PETSC_EXTERN void      SLEPC_BLASLAPACKREAL(lag2,LAG2) (PetscReal*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscReal*,PetscReal*,PetscReal*,PetscReal*,PetscReal*,PetscReal*);
PETSC_EXTERN void      SLEPC_BLASLAPACKREAL(lasv2,LASV2) (PetscReal*,PetscReal*,PetscReal*,PetscReal*,PetscReal*,PetscReal*,PetscReal*,PetscReal*,PetscReal*);
PETSC_EXTERN void      SLEPC_BLASLAPACKREAL(lartg,LARTG) (PetscReal*,PetscReal*,PetscReal*,PetscReal*,PetscReal*);
PETSC_EXTERN void      SLEPC_BLASLAPACKREAL(laln2,LALN2) (PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscReal*,PetscReal*,PetscReal*,PetscBLASInt*,PetscReal*,PetscReal*,PetscReal*,PetscBLASInt*,PetscReal*,PetscReal*,PetscReal*,PetscBLASInt*,PetscReal*,PetscReal*,PetscBLASInt*);
PETSC_EXTERN void      SLEPC_BLASLAPACKREAL(laed4,LAED4) (PetscBLASInt*,PetscBLASInt*,PetscReal*,PetscReal*,PetscReal*,PetscReal*,PetscReal*,PetscBLASInt*);
PETSC_EXTERN void      SLEPC_BLASLAPACKREAL(lamrg,LAMRG) (PetscBLASInt*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN SlepcLRT  SLEPC_BLASLAPACKREAL(lapy2,LAPY2) (PetscReal*,PetscReal*);
PETSC_EXTERN void      BLASrot_(PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscReal*);
#if !defined(PETSC_USE_COMPLEX)
PETSC_EXTERN void      SLEPC_BLASLAPACK(tgexc,TGEXC) (PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN void      SLEPC_BLASLAPACK(orghr,ORGHR) (PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN void      SLEPC_BLASLAPACK(geqp3,GEQP3) (PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
#else
PETSC_EXTERN void      SLEPC_BLASLAPACK(tgexc,TGEXC) (PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN void      SLEPC_BLASLAPACK(unghr,UNGHR) (PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN void      SLEPC_BLASLAPACK(geqp3,GEQP3) (PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscBLASInt*);
#endif

/* LAPACK functions with string parameters */
PETSC_EXTERN SlepcLRT  SLEPC_BLASLAPACK(lanhs,LANHS) (const char*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscBLASInt);
PETSC_EXTERN SlepcLRT  SLEPC_BLASLAPACK(lange,LANGE) (const char*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscBLASInt);
PETSC_EXTERN SlepcLRT  SLEPC_BLASLAPACK(pbtrf,PBTRF) (const char*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt);
PETSC_EXTERN void      SLEPC_BLASLAPACK(larf,LARF) (const char*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt);
PETSC_EXTERN void      SLEPC_BLASLAPACK(trmm,TRMM) (const char*,const char*,const char*,const char*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt,PetscBLASInt,PetscBLASInt,PetscBLASInt);
PETSC_EXTERN void      SLEPC_BLASLAPACK(lacpy,LACPY) (const char*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt);
PETSC_EXTERN SlepcLRT  SLEPC_BLASLAPACK(lansy,LANSY) (const char*,const char*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscBLASInt,PetscBLASInt);
PETSC_EXTERN void      SLEPC_BLASLAPACK(laset,LASET) (const char*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt);
PETSC_EXTERN void      SLEPC_BLASLAPACK(trsyl,TRSYL) (const char*,const char*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscBLASInt,PetscBLASInt);
PETSC_EXTERN void      SLEPC_BLASLAPACK(trtri,TRTRI) (const char*,const char*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt,PetscBLASInt);

PETSC_EXTERN void      SLEPC_BLASLAPACKREAL(stevr,STEVR) (const char*,const char*,PetscBLASInt*,PetscReal*,PetscReal*,PetscReal*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscReal*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt,PetscBLASInt);
PETSC_EXTERN void      SLEPC_BLASLAPACKREAL(bdsdc,BDSDC) (const char*,const char*,PetscBLASInt*,PetscReal*,PetscReal*,PetscReal*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt,PetscBLASInt);
PETSC_EXTERN SlepcLRT  SLEPC_BLASLAPACKREAL(lamch,LAMCH) (const char*,PetscBLASInt);
PETSC_EXTERN SlepcLRT  SLEPC_BLASLAPACKREAL(lamc3,LAMC3) (PetscReal*,PetscReal*);

#if !defined(PETSC_USE_COMPLEX)
PETSC_EXTERN void      SLEPC_BLASLAPACK(ggevx,GGEVX) (const char*,const char*,const char*,const char*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt,PetscBLASInt,PetscBLASInt,PetscBLASInt);
PETSC_EXTERN void      SLEPC_BLASLAPACK(ggev,GGEV) (const char*,const char*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt,PetscBLASInt);
PETSC_EXTERN void      SLEPC_BLASLAPACK(syevr,SYEVR) (const char*,const char*,const char*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt,PetscBLASInt,PetscBLASInt);
PETSC_EXTERN void      SLEPC_BLASLAPACK(syevd,SYEVD) (const char*,const char*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt,PetscBLASInt);
PETSC_EXTERN void      SLEPC_BLASLAPACK(sygvd,SYGVD) (PetscBLASInt*,const char*,const char*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt,PetscBLASInt);
PETSC_EXTERN void      SLEPC_BLASLAPACK(ormlq,ORMLQ) (const char*,const char*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt,PetscBLASInt);
PETSC_EXTERN void      SLEPC_BLASLAPACK(orgtr,ORGTR) (const char*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt);
PETSC_EXTERN void      SLEPC_BLASLAPACK(sytrd,SYTRD) (const char*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscReal*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt);
PETSC_EXTERN void      SLEPC_BLASLAPACK(trevc,TREVC) (const char*,const char*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt,PetscBLASInt);
PETSC_EXTERN void      SLEPC_BLASLAPACK(geevx,GEEVX) (const char*,const char*,const char*,const char*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt,PetscBLASInt,PetscBLASInt,PetscBLASInt);
PETSC_EXTERN void      SLEPC_BLASLAPACK(gees,GEES) (const char*,const char*,PetscBLASInt(*)(PetscReal,PetscReal),PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt,PetscBLASInt);
PETSC_EXTERN void      SLEPC_BLASLAPACK(trexc,TREXC) (const char*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt);
PETSC_EXTERN void      SLEPC_BLASLAPACK(gesdd,GESDD) (const char*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt);
PETSC_EXTERN void      SLEPC_BLASLAPACK(tgevc,TGEVC) (const char*,const char*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt,PetscBLASInt);
PETSC_EXTERN void      SLEPC_BLASLAPACK(hsein,HSEIN) (const char*,const char*,const char*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt,PetscBLASInt,PetscBLASInt);
PETSC_EXTERN void      SLEPC_BLASLAPACK(stedc,STEDC) (const char*,PetscBLASInt*,PetscReal*,PetscReal*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt);
PETSC_EXTERN void      SLEPC_BLASLAPACK(lascl,LASCL) (const char*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt);
#else
PETSC_EXTERN void      SLEPC_BLASLAPACK(ggevx,GGEVX) (const char*,const char*,const char*,const char*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*, PetscBLASInt*,PetscReal*,PetscReal*,PetscReal*,PetscReal*,PetscReal*,PetscReal*,PetscScalar*, PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt,PetscBLASInt,PetscBLASInt,PetscBLASInt);
PETSC_EXTERN void      SLEPC_BLASLAPACK(ggev,GGEV) (const char*,const char*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscBLASInt,PetscBLASInt);
PETSC_EXTERN void      SLEPC_BLASLAPACK(heevr,HEEVR) (const char*,const char*,const char*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscReal*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt,PetscBLASInt,PetscBLASInt);
PETSC_EXTERN void      SLEPC_BLASLAPACK(heevd,HEEVD) (const char*,const char*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt,PetscBLASInt);
PETSC_EXTERN void      SLEPC_BLASLAPACK(hegvd,HEGVD) (PetscBLASInt*,const char*,const char*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt,PetscBLASInt);
PETSC_EXTERN void      SLEPC_BLASLAPACK(unmlq,UNMLQ) (const char*,const char*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt,PetscBLASInt);
PETSC_EXTERN void      SLEPC_BLASLAPACK(ungtr,UNGTR) (const char*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt);
PETSC_EXTERN void      SLEPC_BLASLAPACK(hetrd,HETRD) (const char*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscReal*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt);
PETSC_EXTERN void      SLEPC_BLASLAPACK(trevc,TREVC) (const char*,const char*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscReal*,PetscBLASInt*,PetscBLASInt,PetscBLASInt);
PETSC_EXTERN void      SLEPC_BLASLAPACK(geevx,GEEVX) (const char*,const char*,const char*,const char*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscReal*,PetscReal*,PetscReal*,PetscReal*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscBLASInt,PetscBLASInt,PetscBLASInt,PetscBLASInt);
PETSC_EXTERN void      SLEPC_BLASLAPACK(gees,GEES) (const char*,const char*,PetscBLASInt(*)(PetscScalar),PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt,PetscBLASInt);
PETSC_EXTERN void      SLEPC_BLASLAPACK(trexc,TREXC) (const char*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt);
PETSC_EXTERN void      SLEPC_BLASLAPACK(gesdd,GESDD) (const char*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt);
PETSC_EXTERN void      SLEPC_BLASLAPACK(tgevc,TGEVC) (const char*,const char*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscReal*,PetscBLASInt*,PetscBLASInt,PetscBLASInt);
PETSC_EXTERN void      SLEPC_BLASLAPACK(hsein,HSEIN) (const char*,const char*,const char*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt,PetscBLASInt,PetscBLASInt);
PETSC_EXTERN void      SLEPC_BLASLAPACK(stedc,STEDC) (const char*,PetscBLASInt*,PetscReal*,PetscReal*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt);
PETSC_EXTERN void      SLEPC_BLASLAPACK(lascl,LASCL) (const char*,PetscBLASInt*,PetscBLASInt*,PetscReal*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt);
#endif

#else /* PETSC_BLASLAPACK_STDCALL */

/* LAPACK functions without string parameters */
PETSC_EXTERN void PETSC_STDCALL SLEPC_BLASLAPACK(laev2,LAEV2) (PetscScalar*,PetscScalar*,PetscScalar*,PetscReal*,PetscReal*,PetscReal*,PetscScalar*);
PETSC_EXTERN void PETSC_STDCALL SLEPC_BLASLAPACK(gehrd,GEHRD) (PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN void PETSC_STDCALL SLEPC_BLASLAPACK(gelqf,GELQF) (PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN void PETSC_STDCALL SLEPC_BLASLAPACK(larfg,LARFG) (PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*);
PETSC_EXTERN void PETSC_STDCALL SLEPC_BLASLAPACKREAL(lag2,LAG2) (PetscReal*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscReal*,PetscReal*,PetscReal*,PetscReal*,PetscReal*,PetscReal*);
PETSC_EXTERN void PETSC_STDCALL SLEPC_BLASLAPACKREAL(lasv2,LASV2) (PetscReal*,PetscReal*,PetscReal*,PetscReal*,PetscReal*,PetscReal*,PetscReal*,PetscReal*,PetscReal*);
PETSC_EXTERN void PETSC_STDCALL SLEPC_BLASLAPACKREAL(lartg,LARTG) (PetscReal*,PetscReal*,PetscReal*,PetscReal*,PetscReal*);
PETSC_EXTERN void PETSC_STDCALL SLEPC_BLASLAPACKREAL(laln2,LALN2) (PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscReal*,PetscReal*,PetscReal*,PetscBLASInt*,PetscReal*,PetscReal*,PetscReal*,PetscBLASInt*,PetscReal*,PetscReal*,PetscReal*,PetscBLASInt*,PetscReal*,PetscReal*,PetscBLASInt*);
PETSC_EXTERN void PETSC_STDCALL SLEPC_BLASLAPACKREAL(laed4,LAED4) (PetscBLASInt*,PetscBLASInt*,PetscReal*,PetscReal*,PetscReal*,PetscReal*,PetscReal*,PetscBLASInt*);
PETSC_EXTERN void PETSC_STDCALL SLEPC_BLASLAPACKREAL(lamrg,LAMRG) (PetscBLASInt*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN SlepcLRT  PETSC_STDCALL SLEPC_BLASLAPACKREAL(lapy2,LAPY2) (PetscReal*,PetscReal*);
PETSC_EXTERN void PETSC_STDCALL BLASrot_(PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscReal*);
#if !defined(PETSC_USE_COMPLEX)
PETSC_EXTERN void PETSC_STDCALL SLEPC_BLASLAPACK(tgexc,TGEXC) (PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN void PETSC_STDCALL SLEPC_BLASLAPACK(orghr,ORGHR) (PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN void PETSC_STDCALL SLEPC_BLASLAPACK(geqp3,GEQP3) (PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
#else
PETSC_EXTERN void PETSC_STDCALL SLEPC_BLASLAPACK(tgexc,TGEXC) (PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN void PETSC_STDCALL SLEPC_BLASLAPACK(unghr,UNGHR) (PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN void PETSC_STDCALL SLEPC_BLASLAPACK(geqp3,GEQP3) (PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscBLASInt*);
#endif

/* LAPACK functions with string parameters */
PETSC_EXTERN SlepcLRT  PETSC_STDCALL SLEPC_BLASLAPACK(lanhs,LANHS) (const char*,PetscBLASInt,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*);
PETSC_EXTERN SlepcLRT  PETSC_STDCALL SLEPC_BLASLAPACK(lange,LANGE) (const char*,PetscBLASInt,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*);
PETSC_EXTERN SlepcLRT  PETSC_STDCALL SLEPC_BLASLAPACK(pbtrf,PBTRF) (const char*,PetscBLASInt,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN void PETSC_STDCALL SLEPC_BLASLAPACK(larf,LARF) (const char*,PetscBLASInt,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*);
PETSC_EXTERN void      SLEPC_BLASLAPACK(trmm,TRMM) (const char*,PetscBLASInt,const char*,PetscBLASInt,const char*,PetscBLASInt,const char*,PetscBLASInt,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*);
PETSC_EXTERN void PETSC_STDCALL SLEPC_BLASLAPACK(lacpy,LACPY) (const char*,PetscBLASInt,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*);
PETSC_EXTERN SlepcLRT  PETSC_STDCALL SLEPC_BLASLAPACK(lansy,LANSY) (const char*,PetscBLASInt,const char*,PetscBLASInt,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*);
PETSC_EXTERN void PETSC_STDCALL SLEPC_BLASLAPACK(laset,LASET) (const char*,PetscBLASInt,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscScalar*,PetscBLASInt*);
PETSC_EXTERN void PETSC_STDCALL SLEPC_BLASLAPACK(trsyl,TRSYL) (const char*,PetscBLASInt,const char*,PetscBLASInt,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscBLASInt*);
PETSC_EXTERN void PETSC_STDCALL SLEPC_BLASLAPACK(trtri,TRTRI) (const char*,PetscBLASInt,const char*,PetscBLASInt,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);

PETSC_EXTERN void PETSC_STDCALL SLEPC_BLASLAPACKREAL(stevr,STEVR) (const char*,PetscBLASInt,const char*,PetscBLASInt,PetscBLASInt*,PetscReal*,PetscReal*,PetscReal*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscReal*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN void PETSC_STDCALL SLEPC_BLASLAPACKREAL(bdsdc,BDSDC) (const char*,PetscBLASInt,const char*,PetscBLASInt,PetscBLASInt*,PetscReal*,PetscReal*,PetscReal*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN SlepcLRT  SLEPC_BLASLAPACKREAL(lamch,LAMCH) (const char*,PetscBLASInt);
PETSC_EXTERN SlepcLRT  SLEPC_BLASLAPACKREAL(lamc3,LAMC3) (PetscReal*,PetscReal*);

#if !defined(PETSC_USE_COMPLEX)
PETSC_EXTERN void PETSC_STDCALL SLEPC_BLASLAPACK(ggevx,GGEVX) (const char*,PetscBLASInt,const char*,PetscBLASInt,const char*,PetscBLASInt,const char*,PetscBLASInt,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN void PETSC_STDCALL SLEPC_BLASLAPACK(ggev,GGEV) (const char*,PetscBLASInt,const char*,PetscBLASInt,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN void PETSC_STDCALL SLEPC_BLASLAPACK(syevr,SYEVR) (const char*,PetscBLASInt,const char*,PetscBLASInt,const char*,PetscBLASInt,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN void PETSC_STDCALL SLEPC_BLASLAPACK(syevd,SYEVD) (const char*,PetscBLASInt,const char*,PetscBLASInt,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN void PETSC_STDCALL SLEPC_BLASLAPACK(sygvd,SYGVD) (PetscBLASInt*,const char*,PetscBLASInt,const char*,PetscBLASInt,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN void PETSC_STDCALL SLEPC_BLASLAPACK(ormlq,ORMLQ) (const char*,PetscBLASInt,const char*,PetscBLASInt,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN void PETSC_STDCALL SLEPC_BLASLAPACK(orgtr,ORGTR) (const char*,PetscBLASInt,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN void PETSC_STDCALL SLEPC_BLASLAPACK(sytrd,SYTRD) (const char*,PetscBLASInt,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscReal*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN void PETSC_STDCALL SLEPC_BLASLAPACK(trevc,TREVC) (const char*,PetscBLASInt,const char*,PetscBLASInt,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*);
PETSC_EXTERN void PETSC_STDCALL SLEPC_BLASLAPACK(geevx,GEEVX) (const char*,PetscBLASInt,const char*,PetscBLASInt,const char*,PetscBLASInt,const char*,PetscBLASInt,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN void PETSC_STDCALL SLEPC_BLASLAPACK(gees,GEES) (const char*,PetscBLASInt,const char*,PetscBLASInt,PetscBLASInt(*)(PetscReal,PetscReal),PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN void PETSC_STDCALL SLEPC_BLASLAPACK(trexc,TREXC) (const char*,PetscBLASInt,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*);
PETSC_EXTERN void PETSC_STDCALL SLEPC_BLASLAPACK(gesdd,GESDD) (const char*,PetscBLASInt,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN void PETSC_STDCALL SLEPC_BLASLAPACK(tgevc,TGEVC) (const char*,PetscBLASInt,const char*,PetscBLASInt,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*);
PETSC_EXTERN void PETSC_STDCALL SLEPC_BLASLAPACK(hsein,HSEIN) (const char*,PetscBLASInt,const char*,PetscBLASInt,const char*,PetscBLASInt,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN void PETSC_STDCALL SLEPC_BLASLAPACK(stedc,STEDC) (const char*,PetscBLASInt,PetscBLASInt*,PetscReal*,PetscReal*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN void PETSC_STDCALL SLEPC_BLASLAPACK(lascl,LASCL) (const char*,PetscBLASInt,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
#else
PETSC_EXTERN void PETSC_STDCALL SLEPC_BLASLAPACK(ggevx,GGEVX) (const char*,PetscBLASInt,const char*,PetscBLASInt,const char*,PetscBLASInt,const char*,PetscBLASInt,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*, PetscBLASInt*,PetscReal*,PetscReal*,PetscReal*,PetscReal*,PetscReal*,PetscReal*,PetscScalar*, PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN void PETSC_STDCALL SLEPC_BLASLAPACK(ggev,GGEV) (const char*,PetscBLASInt,const char*,PetscBLASInt,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscBLASInt*);
PETSC_EXTERN void PETSC_STDCALL SLEPC_BLASLAPACK(heevr,HEEVR) (const char*,PetscBLASInt,const char*,PetscBLASInt,const char*,PetscBLASInt,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscReal*,PetscScalar*,PetscBLASInt*,PetscBLASInt*, PetscScalar*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN void PETSC_STDCALL SLEPC_BLASLAPACK(heevd,HEEVD) (const char*,PetscBLASInt,const char*,PetscBLASInt,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN void PETSC_STDCALL SLEPC_BLASLAPACK(hegvd,HEGVD) (PetscBLASInt*,const char*,PetscBLASInt,const char*,PetscBLASInt,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN void PETSC_STDCALL SLEPC_BLASLAPACK(unmlq,UNMLQ) (const char*,PetscBLASInt,const char*,PetscBLASInt,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN void PETSC_STDCALL SLEPC_BLASLAPACK(ungtr,UNGTR) (const char*,PetscBLASInt,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN void PETSC_STDCALL SLEPC_BLASLAPACK(hetrd,HETRD) (const char*,PetscBLASInt,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscReal*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN void PETSC_STDCALL SLEPC_BLASLAPACK(trevc,TREVC) (const char*,PetscBLASInt,const char*,PetscBLASInt,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscReal*,PetscBLASInt*);
PETSC_EXTERN void PETSC_STDCALL SLEPC_BLASLAPACK(geevx,GEEVX) (const char*,PetscBLASInt,const char*,PetscBLASInt,const char*,PetscBLASInt,const char*,PetscBLASInt,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscReal*,PetscReal*,PetscReal*,PetscReal*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscBLASInt*);
PETSC_EXTERN void PETSC_STDCALL SLEPC_BLASLAPACK(gees,GEES) (const char*,PetscBLASInt,const char*,PetscBLASInt,PetscBLASInt(*)(PetscScalar),PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN void PETSC_STDCALL SLEPC_BLASLAPACK(trexc,TREXC) (const char*,PetscBLASInt,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN void PETSC_STDCALL SLEPC_BLASLAPACK(gesdd,GESDD) (const char*,PetscBLASInt,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN void PETSC_STDCALL SLEPC_BLASLAPACK(tgevc,TGEVC) (const char*,PetscBLASInt,const char*,PetscBLASInt,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscReal*,PetscBLASInt*);
PETSC_EXTERN void PETSC_STDCALL SLEPC_BLASLAPACK(hsein,HSEIN) (const char*,PetscBLASInt,const char*,PetscBLASInt,const char*,PetscBLASInt,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN void PETSC_STDCALL SLEPC_BLASLAPACK(stedc,STEDC) (const char*,PetscBLASInt,PetscBLASInt*,PetscReal*,PetscReal*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN void PETSC_STDCALL SLEPC_BLASLAPACK(lascl,LASCL) (const char*,PetscBLASInt,PetscBLASInt*,PetscBLASInt*,PetscReal*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
#endif

#endif

#endif
