/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2017, Universitat Politecnica de Valencia, Spain

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

/* LAPACK functions without string parameters */
#define LAPACKlaev2_ PETSCBLAS(laev2,LAEV2)
#define LAPACKgehrd_ PETSCBLAS(gehrd,GEHRD)
#define LAPACKgelqf_ PETSCBLAS(gelqf,GELQF)
#define LAPACKgeqp3_ PETSCBLAS(geqp3,GEQP3)
#define LAPACKtgexc_ PETSCBLAS(tgexc,TGEXC)
#define LAPACKlarfg_ PETSCBLAS(larfg,LARFG)
#define LAPACKlag2_  PETSCBLASREAL(lag2,LAG2)
#define LAPACKlasv2_ PETSCBLASREAL(lasv2,LASV2)
#define LAPACKlartg_ PETSCBLASREAL(lartg,LARTG)
#define LAPACKlaln2_ PETSCBLASREAL(laln2,LALN2)
#define LAPACKlaed4_ PETSCBLASREAL(laed4,LAED4)
#define LAPACKlamrg_ PETSCBLASREAL(lamrg,LAMRG)
#define LAPACKlapy2_ PETSCBLASREAL(lapy2,LAPY2)
#if !defined(PETSC_USE_COMPLEX)
#define LAPACKorghr_ PETSCBLAS(orghr,ORGHR)
#else
#define LAPACKorghr_ PETSCBLAS(unghr,UNGHR)
#endif
/* the next one needs a special treatment due to the special names:
   srot, drot, csrot, zdrot */
#if !defined(PETSC_USE_COMPLEX)
#define BLASrot_     PETSCBLAS(rot,ROT)
#else
#if defined(PETSC_USE_REAL_SINGLE)
#define BLASrot_     PETSCBLAS(srot,SROT)
#elif defined(PETSC_USE_REAL___FLOAT128)
#define BLASrot_     PETSCBLAS(qrot,QROT)
#else
#define BLASrot_     PETSCBLAS(drot,DROT)
#endif
#endif

/* LAPACK functions with string parameters */
#if !defined(PETSC_BLASLAPACK_STDCALL)

/* same name for real and complex */
#define LAPACKlanhs_(a,b,c,d,e) PETSCBLAS(lanhs,LANHS) ((a),(b),(c),(d),(e),1)
#define LAPACKlange_(a,b,c,d,e,f) PETSCBLAS(lange,LANGE) ((a),(b),(c),(d),(e),(f),1)
#define LAPACKggevx_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z,aa,ab,ac) PETSCBLAS(ggevx,GGEVX) ((a),(b),(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),(o),(p),(q),(r),(s),(t),(u),(v),(w),(x),(y),(z),(aa),(ab),(ac),1,1,1,1)
#define LAPACKggev_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q) PETSCBLAS(ggev,GGEV) ((a),(b),(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),(o),(p),(q),1,1)
#define LAPACKpbtrf_(a,b,c,d,e,f) PETSCBLAS(pbtrf,PBTRF) ((a),(b),(c),(d),(e),(f),1)
#define LAPACKlarf_(a,b,c,d,e,f,g,h,i) PETSCBLAS(larf,LARF) ((a),(b),(c),(d),(e),(f),(g),(h),(i),1)
#define BLAStrmm_(a,b,c,d,e,f,g,h,i,j,k) PETSCBLAS(trmm,TRMM) ((a),(b),(c),(d),(e),(f),(g),(h),(i),(j),(k),1,1,1,1)
#define LAPACKlacpy_(a,b,c,d,e,f,g) PETSCBLAS(lacpy,LACPY) ((a),(b),(c),(d),(e),(f),(g),1)
#define LAPACKlascl_(a,b,c,d,e,f,g,h,i,j) PETSCBLAS(lascl,LASCL) ((a),(b),(c),(d),(e),(f),(g),(h),(i),(j),1)
#define LAPACKlansy_(a,b,c,d,e,f) PETSCBLAS(lansy,LANSY) ((a),(b),(c),(d),(e),(f),1,1)
#define LAPACKlaset_(a,b,c,d,e,f,g) PETSCBLAS(laset,LASET) ((a),(b),(c),(d),(e),(f),(g),1)
#define LAPACKtrsyl_(a,b,c,d,e,f,g,h,i,j,k,l,m) PETSCBLAS(trsyl,TRSYL) ((a),(b),(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),1,1)
#define LAPACKtrtri_(a,b,c,d,e,f) PETSCBLAS(trtri,TRTRI) ((a),(b),(c),(d),(e),(f),1,1)
/* subroutines in which we use only the real version, do not care whether they have different name */
#define LAPACKstevr_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t) PETSCBLASREAL(stevr,STEVR) ((a),(b),(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),(o),(p),(q),(r),(s),(t),1,1)
#define LAPACKbdsdc_(a,b,c,d,e,f,g,h,i,j,k,l,m,n) PETSCBLASREAL(bdsdc,BDSDC) ((a),(b),(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),1,1)
#define LAPACKlamch_(a) PETSCBLASREAL(lamch,LAMCH) ((a),1)
#define LAPACKlamc3_(a,b) PETSCBLASREAL(lamc3,LAMC3) ((a),(b))

#if !defined(PETSC_USE_COMPLEX)
/* different name or signature, real */
#define LAPACKsyevr_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u) PETSCBLAS(syevr,SYEVR) ((a),(b),(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),(o),(p),(q),(r),(s),(t),(u),1,1,1)
#define LAPACKsyevd_(a,b,c,d,e,f,g,h,i,j,k) PETSCBLAS(syevd,SYEVD) ((a),(b),(c),(d),(e),(f),(g),(h),(i),(j),(k),1,1)
#define LAPACKsygvd_(a,b,c,d,e,f,g,h,i,j,k,l,m,n)  PETSCBLAS(sygvd,SYGVD) ((a),(b),(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),1,1)
#define LAPACKormlq_(a,b,c,d,e,f,g,h,i,j,k,l,m) PETSCBLAS(ormlq,ORMLQ) ((a),(b),(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),1,1)
#define LAPACKorgtr_(a,b,c,d,e,f,g,h) PETSCBLAS(orgtr,ORGTR) ((a),(b),(c),(d),(e),(f),(g),(h),1)
#define LAPACKsytrd_(a,b,c,d,e,f,g,h,i,j) PETSCBLAS(sytrd,SYTRD) ((a),(b),(c),(d),(e),(f),(g),(h),(i),(j),1)
#define LAPACKtrevc_(a,b,c,d,e,f,g,h,i,j,k,l,m,n) PETSCBLAS(trevc,TREVC) ((a),(b),(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),1,1)
#define LAPACKgeevx_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w) PETSCBLAS(geevx,GEEVX) ((a),(b),(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),(o),(p),(q),(r),(s),(t),(u),(v),(w),1,1,1,1)
#define LAPACKgees_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o) PETSCBLAS(gees,GEES) ((a),(b),(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),(o),1,1)
#define LAPACKtrexc_(a,b,c,d,e,f,g,h,i,j) PETSCBLAS(trexc,TREXC) ((a),(b),(c),(d),(e),(f),(g),(h),(i),(j),1)
#define LAPACKgesdd_(a,b,c,d,e,f,g,h,i,j,k,l,m,n) PETSCBLAS(gesdd,GESDD) ((a),(b),(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),1)
#define LAPACKtgevc_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p) PETSCBLAS(tgevc,TGEVC) ((a),(b),(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),(o),(p),1,1)
#define LAPACKhsein_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s) PETSCBLAS(hsein,HSEIN) ((a),(b),(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),(o),(p),(q),(r),(s),1,1,1)
#define LAPACKstedc_(a,b,c,d,e,f,g,h,i,j,k) PETSCBLAS(stedc,STEDC) ((a),(b),(c),(d),(e),(f),(g),(h),(i),(j),(k),1)
#else
/* different name or signature, complex */
#define LAPACKsyevr_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w) PETSCBLAS(heevr,HEEVR) ((a),(b),(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),(o),(p),(q),(r),(s),(t),(u),(v),(w),1,1,1)
#define LAPACKsyevd_(a,b,c,d,e,f,g,h,i,j,k,l,m) PETSCBLAS(heevd,HEEVD) ((a),(b),(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),1,1)
#define LAPACKsygvd_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p) PETSCBLAS(hegvd,HEGVD) ((a),(b),(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),(o),(p),1,1)
#define LAPACKormlq_(a,b,c,d,e,f,g,h,i,j,k,l,m) PETSCBLAS(unmlq,UNMLQ) ((a),(b),(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),1,1)
#define LAPACKorgtr_(a,b,c,d,e,f,g,h) PETSCBLAS(ungtr,UNGTR) ((a),(b),(c),(d),(e),(f),(g),(h),1)
#define LAPACKsytrd_(a,b,c,d,e,f,g,h,i,j) PETSCBLAS(hetrd,HETRD) ((a),(b),(c),(d),(e),(f),(g),(h),(i),(j),1)
#define LAPACKtrevc_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o) PETSCBLAS(trevc,TREVC) ((a),(b),(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),(o),1,1)
#define LAPACKgeevx_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v) PETSCBLAS(geevx,GEEVX) ((a),(b),(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),(o),(p),(q),(r),(s),(t),(u),(v),1,1,1,1)
#define LAPACKgees_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o) PETSCBLAS(gees,GEES) ((a),(b),(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),(o),1,1)
#define LAPACKtrexc_(a,b,c,d,e,f,g,h,i) PETSCBLAS(trexc,TREXC) ((a),(b),(c),(d),(e),(f),(g),(h),(i),1)
#define LAPACKgesdd_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o) PETSCBLAS(gesdd,GESDD) ((a),(b),(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),(o),1)
#define LAPACKtgevc_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q) PETSCBLAS(tgevc,TGEVC) ((a),(b),(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),(o),(p),(q),1,1)
#define LAPACKhsein_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s) PETSCBLAS(hsein,HSEIN) ((a),(b),(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),(o),(p),(q),(r),(s),1,1,1)
#define LAPACKstedc_(a,b,c,d,e,f,g,h,i,j,k,l,m) PETSCBLAS(stedc,STEDC) ((a),(b),(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),1)
#endif

#else /* PETSC_BLASLAPACK_STDCALL */

/* same name for real and complex */
#define LAPACKlanhs_(a,b,c,d,e) PETSCBLAS(lanhs,LANHS) ((a),1,(b),(c),(d),(e))
#define LAPACKlange_(a,b,c,d,e,f) PETSCBLAS(lange,LANGE) ((a),1,(b),(c),(d),(e),(f))
#define LAPACKggevx_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z,aa,ab,ac) PETSCBLAS(ggevx,GGEVX) ((a),1,(b),1,(c),1,(d),1,(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),(o),(p),(q),(r),(s),(t),(u),(v),(w),(x),(y),(z),(aa),(ab),(ac))
#define LAPACKggev_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q) PETSCBLAS(ggev,GGEV) ((a),1,(b),1,(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),(o),(p),(q))
#define LAPACKpbtrf_(a,b,c,d,e,f) PETSCBLAS(pbtrf,PBTRF) ((a),1,(b),(c),(d),(e),(f))
#define LAPACKlarf_(a,b,c,d,e,f,g,h,i) PETSCBLAS(larf,LARF) ((a),1,(b),(c),(d),(e),(f),(g),(h),(i))
#define BLAStrmm_(a,b,c,d,e,f,g,h,i,j,k) PETSCBLAS(trmm,TRMM) ((a),1,(b),1,(c),1,(d),1,(e),(f),(g),(h),(i),(j),(k))
#define LAPACKlacpy_(a,b,c,d,e,f,g) PETSCBLAS(lacpy,LACPY) ((a),1,(b),(c),(d),(e),(f),(g))
#define LAPACKlascl_(a,b,c,d,e,f,g,h,i,j) PETSCBLAS(lascl,LASCL) ((a),1,(b),(c),(d),(e),(f),(g),(h),(i),(j))
#define LAPACKlansy_(a,b,c,d,e,f) PETSCBLAS(lansy,LANSY) ((a),1,(b),1,(c),(d),(e),(f))
#define LAPACKlaset_(a,b,c,d,e,f,g) PETSCBLAS(laset,LASET) ((a),1,(b),(c),(d),(e),(f),(g))
#define LAPACKtrsyl_(a,b,c,d,e,f,g,h,i,j,k,l,m) PETSCBLAS(trsyl,TRSYL) ((a),1,(b),1,(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m))
#define LAPACKtrtri_(a,b,c,d,e,f) PETSCBLAS(trtri,TRTRI) ((a),1,(b),1,(c),(d),(e),(f))
/* subroutines in which we use only the real version, do not care whether they have different name */
#define LAPACKstevr_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t) PETSCBLASREAL(stevr,STEVR) ((a),1,(b),1,(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),(o),(p),(q),(r),(s),(t))
#define LAPACKbdsdc_(a,b,c,d,e,f,g,h,i,j,k,l,m,n) PETSCBLASREAL(bdsdc,BDSDC) ((a),1,(b),1,(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n))
#define LAPACKlamch_(a) PETSCBLASREAL(lamch,LAMCH) ((a),1)
#define LAPACKlamc3_(a,b) PETSCBLASREAL(lamc3,LAMC3) ((a),(b))

#if !defined(PETSC_USE_COMPLEX)
/* different name or signature, real */
#define LAPACKsyevr_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u) PETSCBLAS(syevr,SYEVR) ((a),1,(b),1,(c),1,(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),(o),(p),(q),(r),(s),(t),(u))
#define LAPACKsyevd_(a,b,c,d,e,f,g,h,i,j,k) PETSCBLAS(syevd,SYEVD) ((a),1,(b),1,(c),(d),(e),(f),(g),(h),(i),(j),(k))
#define LAPACKsygvd_(a,b,c,d,e,f,g,h,i,j,k,l,m,n)  PETSCBLAS(sygvd,SYGVD) ((a),(b),1,(c),1,(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n))
#define LAPACKormlq_(a,b,c,d,e,f,g,h,i,j,k,l,m) PETSCBLAS(ormlq,ORMLQ) ((a),1,(b),1,(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m))
#define LAPACKorgtr_(a,b,c,d,e,f,g,h) PETSCBLAS(orgtr,ORGTR) ((a),1,(b),(c),(d),(e),(f),(g),(h))
#define LAPACKsytrd_(a,b,c,d,e,f,g,h,i,j) PETSCBLAS(sytrd,SYTRD) ((a),1,(b),(c),(d),(e),(f),(g),(h),(i),(j))
#define LAPACKtrevc_(a,b,c,d,e,f,g,h,i,j,k,l,m,n) PETSCBLAS(trevc,TREVC) ((a),1,(b),1,(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n))
#define LAPACKgeevx_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w) PETSCBLAS(geevx,GEEVX) ((a),1,(b),1,(c),1,(d),1,(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),(o),(p),(q),(r),(s),(t),(u),(v),(w))
#define LAPACKgees_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o) PETSCBLAS(gees,GEES) ((a),1,(b),1,(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),(o))
#define LAPACKtrexc_(a,b,c,d,e,f,g,h,i,j) PETSCBLAS(trexc,TREXC) ((a),1,(b),(c),(d),(e),(f),(g),(h),(i),(j))
#define LAPACKgesdd_(a,b,c,d,e,f,g,h,i,j,k,l,m,n) PETSCBLAS(gesdd,GESDD) ((a),1,(b),(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n))
#define LAPACKtgevc_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p) PETSCBLAS(tgevc,TGEVC) ((a),1,(b),1,(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),(o),(p))
#define LAPACKhsein_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s) PETSCBLAS(hsein,HSEIN) ((a),1,(b),1,(c),1,(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),(o),(p),(q),(r),(s))
#define LAPACKstedc_(a,b,c,d,e,f,g,h,i,j,k) PETSCBLAS(stedc,STEDC) ((a),1,(b),(c),(d),(e),(f),(g),(h),(i),(j),(k))
#else
/* different name or signature, complex */
#define LAPACKsyevr_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w) PETSCBLAS(heevr,HEEVR) ((a),1,(b),1,(c),1,(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),(o),(p),(q),(r),(s),(t),(u),(v),(w))
#define LAPACKsyevd_(a,b,c,d,e,f,g,h,i,j,k,l,m) PETSCBLAS(heevd,HEEVD) ((a),1,(b),1,(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m))
#define LAPACKsygvd_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p) PETSCBLAS(hegvd,HEGVD) ((a),(b),1,(c),1,(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),(o),(p))
#define LAPACKormlq_(a,b,c,d,e,f,g,h,i,j,k,l,m) PETSCBLAS(unmlq,UNMLQ) ((a),1,(b),1,(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m))
#define LAPACKorgtr_(a,b,c,d,e,f,g,h) PETSCBLAS(ungtr,UNGTR) ((a),1,(b),(c),(d),(e),(f),(g),(h))
#define LAPACKsytrd_(a,b,c,d,e,f,g,h,i,j) PETSCBLAS(hetrd,HETRD) ((a),1,(b),(c),(d),(e),(f),(g),(h),(i),(j))
#define LAPACKtrevc_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o) PETSCBLAS(trevc,TREVC) ((a),1,(b),1,(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),(o))
#define LAPACKgeevx_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v) PETSCBLAS(geevx,GEEVX) ((a),1,(b),1,(c),1,(d),1,(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),(o),(p),(q),(r),(s),(t),(u),(v))
#define LAPACKgees_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o) PETSCBLAS(gees,GEES) ((a),1,(b),1,(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),(o))
#define LAPACKtrexc_(a,b,c,d,e,f,g,h,i) PETSCBLAS(trexc,TREXC) ((a),1,(b),(c),(d),(e),(f),(g),(h),(i))
#define LAPACKgesdd_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o) PETSCBLAS(gesdd,GESDD) ((a),1,(b),(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),(o))
#define LAPACKtgevc_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q) PETSCBLAS(tgevc,TGEVC) ((a),1,(b),1,(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),(o),(p),(q))
#define LAPACKhsein_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s) PETSCBLAS(hsein,HSEIN) ((a),1,(b),1,(c),1,(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),(o),(p),(q),(r),(s))
#define LAPACKstedc_(a,b,c,d,e,f,g,h,i,j,k,l,m) PETSCBLAS(stedc,STEDC) ((a),1,(b),(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m))
#endif

#endif

#if !defined(PETSC_BLASLAPACK_STDCALL)

/* LAPACK functions without string parameters */
PETSC_EXTERN void      PETSCBLAS(laev2,LAEV2) (PetscScalar*,PetscScalar*,PetscScalar*,PetscReal*,PetscReal*,PetscReal*,PetscScalar*);
PETSC_EXTERN void      PETSCBLAS(gehrd,GEHRD) (PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN void      PETSCBLAS(gelqf,GELQF) (PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN void      PETSCBLAS(larfg,LARFG) (PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*);
PETSC_EXTERN void      PETSCBLASREAL(lag2,LAG2) (PetscReal*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscReal*,PetscReal*,PetscReal*,PetscReal*,PetscReal*,PetscReal*);
PETSC_EXTERN void      PETSCBLASREAL(lasv2,LASV2) (PetscReal*,PetscReal*,PetscReal*,PetscReal*,PetscReal*,PetscReal*,PetscReal*,PetscReal*,PetscReal*);
PETSC_EXTERN void      PETSCBLASREAL(lartg,LARTG) (PetscReal*,PetscReal*,PetscReal*,PetscReal*,PetscReal*);
PETSC_EXTERN void      PETSCBLASREAL(laln2,LALN2) (PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscReal*,PetscReal*,PetscReal*,PetscBLASInt*,PetscReal*,PetscReal*,PetscReal*,PetscBLASInt*,PetscReal*,PetscReal*,PetscReal*,PetscBLASInt*,PetscReal*,PetscReal*,PetscBLASInt*);
PETSC_EXTERN void      PETSCBLASREAL(laed4,LAED4) (PetscBLASInt*,PetscBLASInt*,PetscReal*,PetscReal*,PetscReal*,PetscReal*,PetscReal*,PetscBLASInt*);
PETSC_EXTERN void      PETSCBLASREAL(lamrg,LAMRG) (PetscBLASInt*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN SlepcLRT  PETSCBLASREAL(lapy2,LAPY2) (PetscReal*,PetscReal*);
PETSC_EXTERN void      BLASrot_(PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscReal*);
#if !defined(PETSC_USE_COMPLEX)
PETSC_EXTERN void      PETSCBLAS(tgexc,TGEXC) (PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN void      PETSCBLAS(orghr,ORGHR) (PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN void      PETSCBLAS(geqp3,GEQP3) (PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
#else
PETSC_EXTERN void      PETSCBLAS(tgexc,TGEXC) (PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN void      PETSCBLAS(unghr,UNGHR) (PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN void      PETSCBLAS(geqp3,GEQP3) (PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscBLASInt*);
#endif

/* LAPACK functions with string parameters */
PETSC_EXTERN SlepcLRT  PETSCBLAS(lanhs,LANHS) (const char*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscBLASInt);
PETSC_EXTERN SlepcLRT  PETSCBLAS(lange,LANGE) (const char*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscBLASInt);
PETSC_EXTERN SlepcLRT  PETSCBLAS(pbtrf,PBTRF) (const char*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt);
PETSC_EXTERN void      PETSCBLAS(larf,LARF) (const char*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt);
PETSC_EXTERN void      PETSCBLAS(trmm,TRMM) (const char*,const char*,const char*,const char*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt,PetscBLASInt,PetscBLASInt,PetscBLASInt);
PETSC_EXTERN void      PETSCBLAS(lacpy,LACPY) (const char*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt);
PETSC_EXTERN SlepcLRT  PETSCBLAS(lansy,LANSY) (const char*,const char*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscBLASInt,PetscBLASInt);
PETSC_EXTERN void      PETSCBLAS(laset,LASET) (const char*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt);
PETSC_EXTERN void      PETSCBLAS(trsyl,TRSYL) (const char*,const char*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscBLASInt,PetscBLASInt);
PETSC_EXTERN void      PETSCBLAS(trtri,TRTRI) (const char*,const char*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt,PetscBLASInt);

PETSC_EXTERN void      PETSCBLASREAL(stevr,STEVR) (const char*,const char*,PetscBLASInt*,PetscReal*,PetscReal*,PetscReal*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscReal*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt,PetscBLASInt);
PETSC_EXTERN void      PETSCBLASREAL(bdsdc,BDSDC) (const char*,const char*,PetscBLASInt*,PetscReal*,PetscReal*,PetscReal*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt,PetscBLASInt);
PETSC_EXTERN SlepcLRT  PETSCBLASREAL(lamch,LAMCH) (const char*,PetscBLASInt);
PETSC_EXTERN SlepcLRT  PETSCBLASREAL(lamc3,LAMC3) (PetscReal*,PetscReal*);

#if !defined(PETSC_USE_COMPLEX)
PETSC_EXTERN void      PETSCBLAS(ggevx,GGEVX) (const char*,const char*,const char*,const char*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt,PetscBLASInt,PetscBLASInt,PetscBLASInt);
PETSC_EXTERN void      PETSCBLAS(ggev,GGEV) (const char*,const char*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt,PetscBLASInt);
PETSC_EXTERN void      PETSCBLAS(syevr,SYEVR) (const char*,const char*,const char*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt,PetscBLASInt,PetscBLASInt);
PETSC_EXTERN void      PETSCBLAS(syevd,SYEVD) (const char*,const char*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt,PetscBLASInt);
PETSC_EXTERN void      PETSCBLAS(sygvd,SYGVD) (PetscBLASInt*,const char*,const char*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt,PetscBLASInt);
PETSC_EXTERN void      PETSCBLAS(ormlq,ORMLQ) (const char*,const char*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt,PetscBLASInt);
PETSC_EXTERN void      PETSCBLAS(orgtr,ORGTR) (const char*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt);
PETSC_EXTERN void      PETSCBLAS(sytrd,SYTRD) (const char*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscReal*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt);
PETSC_EXTERN void      PETSCBLAS(trevc,TREVC) (const char*,const char*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt,PetscBLASInt);
PETSC_EXTERN void      PETSCBLAS(geevx,GEEVX) (const char*,const char*,const char*,const char*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt,PetscBLASInt,PetscBLASInt,PetscBLASInt);
PETSC_EXTERN void      PETSCBLAS(gees,GEES) (const char*,const char*,PetscBLASInt(*)(PetscReal,PetscReal),PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt,PetscBLASInt);
PETSC_EXTERN void      PETSCBLAS(trexc,TREXC) (const char*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt);
PETSC_EXTERN void      PETSCBLAS(gesdd,GESDD) (const char*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt);
PETSC_EXTERN void      PETSCBLAS(tgevc,TGEVC) (const char*,const char*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt,PetscBLASInt);
PETSC_EXTERN void      PETSCBLAS(hsein,HSEIN) (const char*,const char*,const char*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt,PetscBLASInt,PetscBLASInt);
PETSC_EXTERN void      PETSCBLAS(stedc,STEDC) (const char*,PetscBLASInt*,PetscReal*,PetscReal*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt);
PETSC_EXTERN void      PETSCBLAS(lascl,LASCL) (const char*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt);
#else
PETSC_EXTERN void      PETSCBLAS(ggevx,GGEVX) (const char*,const char*,const char*,const char*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*, PetscBLASInt*,PetscReal*,PetscReal*,PetscReal*,PetscReal*,PetscReal*,PetscReal*,PetscScalar*, PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt,PetscBLASInt,PetscBLASInt,PetscBLASInt);
PETSC_EXTERN void      PETSCBLAS(ggev,GGEV) (const char*,const char*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscBLASInt,PetscBLASInt);
PETSC_EXTERN void      PETSCBLAS(heevr,HEEVR) (const char*,const char*,const char*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscReal*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt,PetscBLASInt,PetscBLASInt);
PETSC_EXTERN void      PETSCBLAS(heevd,HEEVD) (const char*,const char*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt,PetscBLASInt);
PETSC_EXTERN void      PETSCBLAS(hegvd,HEGVD) (PetscBLASInt*,const char*,const char*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt,PetscBLASInt);
PETSC_EXTERN void      PETSCBLAS(unmlq,UNMLQ) (const char*,const char*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt,PetscBLASInt);
PETSC_EXTERN void      PETSCBLAS(ungtr,UNGTR) (const char*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt);
PETSC_EXTERN void      PETSCBLAS(hetrd,HETRD) (const char*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscReal*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt);
PETSC_EXTERN void      PETSCBLAS(trevc,TREVC) (const char*,const char*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscReal*,PetscBLASInt*,PetscBLASInt,PetscBLASInt);
PETSC_EXTERN void      PETSCBLAS(geevx,GEEVX) (const char*,const char*,const char*,const char*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscReal*,PetscReal*,PetscReal*,PetscReal*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscBLASInt,PetscBLASInt,PetscBLASInt,PetscBLASInt);
PETSC_EXTERN void      PETSCBLAS(gees,GEES) (const char*,const char*,PetscBLASInt(*)(PetscScalar),PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt,PetscBLASInt);
PETSC_EXTERN void      PETSCBLAS(trexc,TREXC) (const char*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt);
PETSC_EXTERN void      PETSCBLAS(gesdd,GESDD) (const char*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt);
PETSC_EXTERN void      PETSCBLAS(tgevc,TGEVC) (const char*,const char*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscReal*,PetscBLASInt*,PetscBLASInt,PetscBLASInt);
PETSC_EXTERN void      PETSCBLAS(hsein,HSEIN) (const char*,const char*,const char*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt,PetscBLASInt,PetscBLASInt);
PETSC_EXTERN void      PETSCBLAS(stedc,STEDC) (const char*,PetscBLASInt*,PetscReal*,PetscReal*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt);
PETSC_EXTERN void      PETSCBLAS(lascl,LASCL) (const char*,PetscBLASInt*,PetscBLASInt*,PetscReal*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt);
#endif

#else /* PETSC_BLASLAPACK_STDCALL */

/* LAPACK functions without string parameters */
PETSC_EXTERN void PETSC_STDCALL PETSCBLAS(laev2,LAEV2) (PetscScalar*,PetscScalar*,PetscScalar*,PetscReal*,PetscReal*,PetscReal*,PetscScalar*);
PETSC_EXTERN void PETSC_STDCALL PETSCBLAS(gehrd,GEHRD) (PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN void PETSC_STDCALL PETSCBLAS(gelqf,GELQF) (PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN void PETSC_STDCALL PETSCBLAS(larfg,LARFG) (PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*);
PETSC_EXTERN void PETSC_STDCALL PETSCBLASREAL(lag2,LAG2) (PetscReal*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscReal*,PetscReal*,PetscReal*,PetscReal*,PetscReal*,PetscReal*);
PETSC_EXTERN void PETSC_STDCALL PETSCBLASREAL(lasv2,LASV2) (PetscReal*,PetscReal*,PetscReal*,PetscReal*,PetscReal*,PetscReal*,PetscReal*,PetscReal*,PetscReal*);
PETSC_EXTERN void PETSC_STDCALL PETSCBLASREAL(lartg,LARTG) (PetscReal*,PetscReal*,PetscReal*,PetscReal*,PetscReal*);
PETSC_EXTERN void PETSC_STDCALL PETSCBLASREAL(laln2,LALN2) (PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscReal*,PetscReal*,PetscReal*,PetscBLASInt*,PetscReal*,PetscReal*,PetscReal*,PetscBLASInt*,PetscReal*,PetscReal*,PetscReal*,PetscBLASInt*,PetscReal*,PetscReal*,PetscBLASInt*);
PETSC_EXTERN void PETSC_STDCALL PETSCBLASREAL(laed4,LAED4) (PetscBLASInt*,PetscBLASInt*,PetscReal*,PetscReal*,PetscReal*,PetscReal*,PetscReal*,PetscBLASInt*);
PETSC_EXTERN void PETSC_STDCALL PETSCBLASREAL(lamrg,LAMRG) (PetscBLASInt*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN SlepcLRT  PETSC_STDCALL PETSCBLASREAL(lapy2,LAPY2) (PetscReal*,PetscReal*);
PETSC_EXTERN void PETSC_STDCALL BLASrot_(PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscReal*);
#if !defined(PETSC_USE_COMPLEX)
PETSC_EXTERN void PETSC_STDCALL PETSCBLAS(tgexc,TGEXC) (PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN void PETSC_STDCALL PETSCBLAS(orghr,ORGHR) (PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN void PETSC_STDCALL PETSCBLAS(geqp3,GEQP3) (PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
#else
PETSC_EXTERN void PETSC_STDCALL PETSCBLAS(tgexc,TGEXC) (PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN void PETSC_STDCALL PETSCBLAS(unghr,UNGHR) (PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN void PETSC_STDCALL PETSCBLAS(geqp3,GEQP3) (PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscBLASInt*);
#endif

/* LAPACK functions with string parameters */
PETSC_EXTERN SlepcLRT  PETSC_STDCALL PETSCBLAS(lanhs,LANHS) (const char*,PetscBLASInt,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*);
PETSC_EXTERN SlepcLRT  PETSC_STDCALL PETSCBLAS(lange,LANGE) (const char*,PetscBLASInt,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*);
PETSC_EXTERN SlepcLRT  PETSC_STDCALL PETSCBLAS(pbtrf,PBTRF) (const char*,PetscBLASInt,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN void PETSC_STDCALL PETSCBLAS(larf,LARF) (const char*,PetscBLASInt,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*);
PETSC_EXTERN void      PETSCBLAS(trmm,TRMM) (const char*,PetscBLASInt,const char*,PetscBLASInt,const char*,PetscBLASInt,const char*,PetscBLASInt,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*);
PETSC_EXTERN void PETSC_STDCALL PETSCBLAS(lacpy,LACPY) (const char*,PetscBLASInt,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*);
PETSC_EXTERN SlepcLRT  PETSC_STDCALL PETSCBLAS(lansy,LANSY) (const char*,PetscBLASInt,const char*,PetscBLASInt,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*);
PETSC_EXTERN void PETSC_STDCALL PETSCBLAS(laset,LASET) (const char*,PetscBLASInt,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscScalar*,PetscBLASInt*);
PETSC_EXTERN void PETSC_STDCALL PETSCBLAS(trsyl,TRSYL) (const char*,PetscBLASInt,const char*,PetscBLASInt,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscBLASInt*);
PETSC_EXTERN void PETSC_STDCALL PETSCBLAS(trtri,TRTRI) (const char*,PetscBLASInt,const char*,PetscBLASInt,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);

PETSC_EXTERN void PETSC_STDCALL PETSCBLASREAL(stevr,STEVR) (const char*,PetscBLASInt,const char*,PetscBLASInt,PetscBLASInt*,PetscReal*,PetscReal*,PetscReal*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscReal*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN void PETSC_STDCALL PETSCBLASREAL(bdsdc,BDSDC) (const char*,PetscBLASInt,const char*,PetscBLASInt,PetscBLASInt*,PetscReal*,PetscReal*,PetscReal*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN SlepcLRT  PETSCBLASREAL(lamch,LAMCH) (const char*,PetscBLASInt);
PETSC_EXTERN SlepcLRT  PETSCBLASREAL(lamc3,LAMC3) (PetscReal*,PetscReal*);

#if !defined(PETSC_USE_COMPLEX)
PETSC_EXTERN void PETSC_STDCALL PETSCBLAS(ggevx,GGEVX) (const char*,PetscBLASInt,const char*,PetscBLASInt,const char*,PetscBLASInt,const char*,PetscBLASInt,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN void PETSC_STDCALL PETSCBLAS(ggev,GGEV) (const char*,PetscBLASInt,const char*,PetscBLASInt,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN void PETSC_STDCALL PETSCBLAS(syevr,SYEVR) (const char*,PetscBLASInt,const char*,PetscBLASInt,const char*,PetscBLASInt,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN void PETSC_STDCALL PETSCBLAS(syevd,SYEVD) (const char*,PetscBLASInt,const char*,PetscBLASInt,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN void PETSC_STDCALL PETSCBLAS(sygvd,SYGVD) (PetscBLASInt*,const char*,PetscBLASInt,const char*,PetscBLASInt,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN void PETSC_STDCALL PETSCBLAS(ormlq,ORMLQ) (const char*,PetscBLASInt,const char*,PetscBLASInt,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN void PETSC_STDCALL PETSCBLAS(orgtr,ORGTR) (const char*,PetscBLASInt,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN void PETSC_STDCALL PETSCBLAS(sytrd,SYTRD) (const char*,PetscBLASInt,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscReal*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN void PETSC_STDCALL PETSCBLAS(trevc,TREVC) (const char*,PetscBLASInt,const char*,PetscBLASInt,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*);
PETSC_EXTERN void PETSC_STDCALL PETSCBLAS(geevx,GEEVX) (const char*,PetscBLASInt,const char*,PetscBLASInt,const char*,PetscBLASInt,const char*,PetscBLASInt,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN void PETSC_STDCALL PETSCBLAS(gees,GEES) (const char*,PetscBLASInt,const char*,PetscBLASInt,PetscBLASInt(*)(PetscReal,PetscReal),PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN void PETSC_STDCALL PETSCBLAS(trexc,TREXC) (const char*,PetscBLASInt,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*);
PETSC_EXTERN void PETSC_STDCALL PETSCBLAS(gesdd,GESDD) (const char*,PetscBLASInt,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN void PETSC_STDCALL PETSCBLAS(tgevc,TGEVC) (const char*,PetscBLASInt,const char*,PetscBLASInt,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*);
PETSC_EXTERN void PETSC_STDCALL PETSCBLAS(hsein,HSEIN) (const char*,PetscBLASInt,const char*,PetscBLASInt,const char*,PetscBLASInt,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN void PETSC_STDCALL PETSCBLAS(stedc,STEDC) (const char*,PetscBLASInt,PetscBLASInt*,PetscReal*,PetscReal*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN void PETSC_STDCALL PETSCBLAS(lascl,LASCL) (const char*,PetscBLASInt,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
#else
PETSC_EXTERN void PETSC_STDCALL PETSCBLAS(ggevx,GGEVX) (const char*,PetscBLASInt,const char*,PetscBLASInt,const char*,PetscBLASInt,const char*,PetscBLASInt,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*, PetscBLASInt*,PetscReal*,PetscReal*,PetscReal*,PetscReal*,PetscReal*,PetscReal*,PetscScalar*, PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN void PETSC_STDCALL PETSCBLAS(ggev,GGEV) (const char*,PetscBLASInt,const char*,PetscBLASInt,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscBLASInt*);
PETSC_EXTERN void PETSC_STDCALL PETSCBLAS(heevr,HEEVR) (const char*,PetscBLASInt,const char*,PetscBLASInt,const char*,PetscBLASInt,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscReal*,PetscScalar*,PetscBLASInt*,PetscBLASInt*, PetscScalar*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN void PETSC_STDCALL PETSCBLAS(heevd,HEEVD) (const char*,PetscBLASInt,const char*,PetscBLASInt,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN void PETSC_STDCALL PETSCBLAS(hegvd,HEGVD) (PetscBLASInt*,const char*,PetscBLASInt,const char*,PetscBLASInt,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN void PETSC_STDCALL PETSCBLAS(unmlq,UNMLQ) (const char*,PetscBLASInt,const char*,PetscBLASInt,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN void PETSC_STDCALL PETSCBLAS(ungtr,UNGTR) (const char*,PetscBLASInt,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN void PETSC_STDCALL PETSCBLAS(hetrd,HETRD) (const char*,PetscBLASInt,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscReal*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN void PETSC_STDCALL PETSCBLAS(trevc,TREVC) (const char*,PetscBLASInt,const char*,PetscBLASInt,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscReal*,PetscBLASInt*);
PETSC_EXTERN void PETSC_STDCALL PETSCBLAS(geevx,GEEVX) (const char*,PetscBLASInt,const char*,PetscBLASInt,const char*,PetscBLASInt,const char*,PetscBLASInt,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscReal*,PetscReal*,PetscReal*,PetscReal*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscBLASInt*);
PETSC_EXTERN void PETSC_STDCALL PETSCBLAS(gees,GEES) (const char*,PetscBLASInt,const char*,PetscBLASInt,PetscBLASInt(*)(PetscScalar),PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN void PETSC_STDCALL PETSCBLAS(trexc,TREXC) (const char*,PetscBLASInt,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN void PETSC_STDCALL PETSCBLAS(gesdd,GESDD) (const char*,PetscBLASInt,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN void PETSC_STDCALL PETSCBLAS(tgevc,TGEVC) (const char*,PetscBLASInt,const char*,PetscBLASInt,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscReal*,PetscBLASInt*);
PETSC_EXTERN void PETSC_STDCALL PETSCBLAS(hsein,HSEIN) (const char*,PetscBLASInt,const char*,PetscBLASInt,const char*,PetscBLASInt,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN void PETSC_STDCALL PETSCBLAS(stedc,STEDC) (const char*,PetscBLASInt,PetscBLASInt*,PetscReal*,PetscReal*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN void PETSC_STDCALL PETSCBLAS(lascl,LASCL) (const char*,PetscBLASInt,PetscBLASInt*,PetscBLASInt*,PetscReal*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
#endif

#endif

#endif
