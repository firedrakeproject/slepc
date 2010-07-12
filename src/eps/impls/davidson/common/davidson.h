/*
  SLEPc eigensolver: "davidson"

  Method: General Davidson Method

  References:
    - Ernest R. Davidson. Super-matrix methods. Computer Physics Communications,
      53:49–60, May 1989.

  TODO:
    - If the problem is symmetric, H=V*A*V is symmetric too, we can do some ops:
        *) Use a LAPACK eigensolver that only modify the upper triangular part
           of H, so we can save H in the lower one, and in a vector for the
           diagonal. That expends only 2*n^2 + n of memory.

    - In the interface, support static preconditioners of PETSc (PC) and others
      with shift parameter.

    - (DONE) Start with a krylov subspace, the H matrix will be used as the starting
      projected problem matrix, and A*V <- V*H + f*e_m^T.

    - (DONE) Implements with BLAS3 prods Z <- V*U

    - Implement the orthonormalization routines with BLAS3

    - Implement an interface for memory management

    - Store V, W, AV and BV like [V; W; AV; BV] to make easier the management.

*/


/* 
   Dashboard struct: contains the methods that will be employed and the tunning
   options.
*/

#include "petsc.h"
#include "private/epsimpl.h"
#include "private/stimpl.h"


typedef struct _dvdFunctionList {
  PetscInt (*f)(void*);
  void *d;
  struct _dvdFunctionList *next;
} dvdFunctionList;

typedef PetscInt MatType_t;
#define DVD_MAT_HERMITIAN (1<<1)
#define DVD_MAT_NEG_DEF (1<<2)
#define DVD_MAT_POS_DEF (1<<3)
#define DVD_MAT_SINGULAR (1<<4)
#define DVD_MAT_COMPLEX (1<<5)
#define DVD_MAT_IMPLICIT (1<<6)
#define DVD_MAT_IDENTITY (1<<7)
#define DVD_MAT_DIAG (1<<8)
#define DVD_MAT_TRIANG (1<<9)
#define DVD_MAT_UTRIANG (1<<9)
#define DVD_MAT_LTRIANG (1<<10)
#define DVD_MAT_UNITARY (1<<11)

typedef PetscInt EPType_t;
#define DVD_EP_STD (1<<1)
#define DVD_EP_HERMITIAN (1<<2)

#define DVD_IS(T,P) ((T) & (P))
#define DVD_ISNOT(T,P) (((T) & (P)) ^ (P))

typedef enum {
  DVD_HARM_NONE,
  DVD_HARM_RR,
  DVD_HARM_RRR,
  DVD_HARM_REIGS,
  DVD_HARM_LEIGS
} HarmType_t;

typedef enum {
  DVD_INITV_CLASSIC,
  DVD_INITV_KRYLOV
} InitType_t;

typedef enum {
  DVD_PROJ_KBXX,
  DVD_PROJ_KBXY,
  DVD_PROJ_KBXZ,
  DVD_PROJ_KBXZY
} ProjType_t;

typedef enum {
    DVD_MT_IDENTITY,/* without transformation */
    DVD_MT_pX,       /* using the projected problem eigenvectors */
    DVD_MT_ORTHO    /* using an orthonormal transformation */
} MT_type_t;

typedef enum {
    DVD_ORTHOV_NONE,/* V isn't orthonormalized */
    DVD_ORTHOV_I,   /* V is orthonormalized */     
    DVD_ORTHOV_B    /* V is B-orthonormalized */
} orthoV_type_t;


typedef struct _dvdDashboard {
  /**** Function steps ****/
  /* Initialize V */
  PetscInt (*initV)(struct _dvdDashboard*);
  void *initV_data;
  
  /* Find the approximate eigenpairs from V */
  PetscInt (*calcPairs)(struct _dvdDashboard*);
  void *calcPairs_data;

  /* Test for convergence */
  PetscTruth (*testConv)(struct _dvdDashboard*, PetscScalar eigvr,
       PetscScalar eigvi, PetscReal res, PetscReal *error);
  void *testConv_data;

  /* Number of converged eigenpairs */
  PetscInt nconv;

  /* Number of pairs ready to converge */
  PetscInt npreconv;

  /* Improve the selected eigenpairs */
  PetscInt (*improveX)(struct _dvdDashboard*, Vec *D, PetscInt max_size_D,
                       PetscInt r_s, PetscInt r_e, PetscInt *size_D);
  void *improveX_data;

  /* Check for restarting */
  PetscTruth (*isRestarting)(struct _dvdDashboard*);
  void *isRestarting_data;

  /* Perform restarting */
  PetscInt (*restartV)(struct _dvdDashboard*);
  void *restartV_data;

  /* Update V */
  PetscInt (*updateV)(struct _dvdDashboard*);
  void *updateV_data;

  /**** Problem specification ****/
  Mat A, B;         /* Problem matrices */
  MatType_t sA, sB; /* Matrix specifications */
  EPType_t sEP;     /* Problem specifications */
  PetscInt nev;     /* number of eigenpairs */
  EPSWhich which;   /* spectrum selection */
  PetscTruth
    withTarget;     /* if there is a target */
  PetscScalar
    target[2];         /* target value */
  PetscReal tol;    /* tolerance */
  PetscTruth
    correctXnorm;   /* if true, tol < |r|/|x| */

  /**** Subspaces specification ****/
  Vec *V,           /* searching subspace */
    *W,             /* testing subspace */
    *cX,            /* converged right eigenvectors */
    *cY,            /* converged left eigenvectors */
    *BcX,           /* basis of B*cX */
    *AV,            /* A*V */
    *real_AV,       /* original A*V space */
    *BV,            /* B*V */
    *real_BV;       /* original B*V space */
  PetscInt size_V,  /* size of V */
    size_AV,        /* size of AV */
    size_BV,        /* size of BV */
    size_cX,        /* size of cX */
    size_cY,        /* size of cY */
    size_D,         /* active vectors */
    max_size_V,     /* max size of V */
    max_size_X,     /* max size of X */
    max_size_AV,    /* max size of AV */
    max_size_BV;    /* max size of BV */
  EPS eps;          /* Connection to SLEPc */
 
  /**** Auxiliary space ****/
  Vec *auxV;        /* auxiliary vectors */
  PetscScalar
    *auxS;          /* auxiliary scalars */
  PetscInt
    size_auxV,      /* max size of auxV */
    size_auxS;      /* max size of auxS */

  /**** Eigenvalues and errors ****/
  PetscScalar
    *ceigr, *ceigi, /* converged eigenvalues */
    *eigr, *eigi;   /* current eigenvalues */
  PetscReal
    *nR,            /* residual norm */
    *nX,            /* X norm */
    *errest;        /* relative error eigenpairs */

  /**** Shared function and variables ****/
  PetscInt (*e_Vchanged)(struct _dvdDashboard*, PetscInt s_imm,
                         PetscInt e_imm, PetscInt s_new, PetscInt e_new);
  void *e_Vchanged_data;

  PetscInt (*calcpairs_residual)(struct _dvdDashboard*, PetscInt s, PetscInt e,
                                 Vec *R, PetscScalar *auxS, Vec auxV);
  PetscInt (*calcpairs_selectPairs)(struct _dvdDashboard*, PetscInt n);
  void *calcpairs_residual_data;
  PetscInt (*calcpairs_X)(struct _dvdDashboard*, PetscInt s, PetscInt e,
                          Vec *X);
  PetscInt (*calcpairs_Y)(struct _dvdDashboard*, PetscInt s, PetscInt e,
                          Vec *Y);
  PetscErrorCode (*improvex_precond)(struct _dvdDashboard*, PetscInt i, Vec x,
                  Vec Px);
  void *improvex_precond_data;
  PetscErrorCode (*improvex_jd_proj_uv)(struct _dvdDashboard*, PetscInt i_s,
                                        PetscInt i_e, Vec **u, Vec **v,
                                        Vec **kr, Vec **auxV_,
                                        PetscScalar *theta,
                                        PetscScalar *thetai,
                                        PetscScalar *pX, PetscScalar *pY,
                                        PetscInt ld);
  PetscErrorCode (*improvex_jd_lit)(struct _dvdDashboard*, PetscInt i,
                                    PetscScalar* theta, PetscScalar* thetai,
                                    PetscInt *maxits, PetscReal *tol);
  PetscErrorCode (*calcpairs_W)(struct _dvdDashboard*);
  void *calcpairs_W_data;
  PetscErrorCode (*calcpairs_proj_trans)(struct _dvdDashboard*);
  PetscErrorCode (*calcpairs_eigs_trans)(struct _dvdDashboard*);
  PetscErrorCode (*calcpairs_proj_res)(struct _dvdDashboard*, PetscInt r_s,
                  PetscInt r_e, Vec *R);

  PetscInt (*e_newIteration)(struct _dvdDashboard*);
  void *e_newIteration_data;

  IP ipI;
  IP ipV,           /* orthogonal routine options for V subspace */
    ipW;            /* orthogonal routine options for W subspace */

  dvdFunctionList
    *startList,     /* starting list */
    *endList,       /* ending list */
    *destroyList;   /* destructor list */

  PetscScalar *H,   /* Projected problem matrix A*/
    *real_H,        /* original H */
    *G,             /* Projected problem matrix B*/
    *real_G,        /* original G */
    *pX,            /* projected problem right eigenvectors */
    *pY,            /* projected problem left eigenvectors */
    *MTX,           /* right transformation matrix */
    *MTY,           /* left transformation matrix */
    *S,             /* first Schur matrix, S = pY'*H*pX */
    *T,             /* second Schur matrix, T = pY'*G*pX */
    *cS,            /* first Schur matrix of converged pairs */
    *cT;            /* second Schur matrix of converged pairs */
  MatType_t
    pX_type,        /* pX properties */
    pY_type,        /* pY properties */
    sH,             /* H properties */
    sG;             /* G properties */
  PetscInt ldH,     /* leading dimension of H */
    ldpX,           /* leading dimension of pX */
    ldpY,           /* leading dimension of pY */
    ldMTX,          /* leading dimension of MTX */
    ldMTY,          /* leading dimension of MTY */
    ldS,            /* leading dimension of S */
    ldT,            /* leading dimension of T */
    ldcS,           /* leading dimension of cS */
    ldcT,           /* leading dimension of cT */
    size_H,         /* rows and columns in H */
    size_G,         /* rows and columns in G */
    size_MT;        /* rows in MT */

  PetscInt V_imm_s,
    V_imm_e,        /* unchanged V columns, V_imm_s:V_imm_e-1 */
    V_tra_s,
    V_tra_e,        /* V(V_tra_e:) = V*MT(V_tra_s:V_tra_e-1) */
    V_new_s,
    V_new_e;        /* added to V the columns V_new_s:V_new_e */

  MT_type_t MT_type;
  orthoV_type_t orthoV_type;

  PetscRandom rand; /* random seed */
  void* prof_data;  /* profiler data */
} dvdDashboard;

#define DVD_FL_ADD(list, fun) { \
  dvdFunctionList *fl=(list); \
  PetscErrorCode ierr; \
  ierr = PetscMalloc(sizeof(dvdFunctionList), &(list)); CHKERRQ(ierr); \
  (list)->f = (PetscInt(*)(void*))(fun); \
  (list)->next = fl; }

#define DVD_FL_CALL(list, arg0) { \
  dvdFunctionList *fl; \
  for(fl=(list); fl; fl=fl->next) (*(dvdCallback)fl->f)((arg0)); }

#define DVD_FL_DEL(list) { \
  dvdFunctionList *fl=(list), *oldfl; \
  PetscErrorCode ierr; \
  while(fl) { \
    oldfl = fl; fl = fl->next; ierr = PetscFree(oldfl); CHKERRQ(ierr); }}

/*
  The blackboard configuration structure: saves information about the memory
  and other requirements
*/
typedef struct {
  PetscInt max_size_V,  /* max size of V */
    max_size_X,         /* max size of X */
    max_size_oldX,      /* max size of oldX */
    max_size_auxV,      /* max size of auxiliary vecs */
    max_size_auxS,      /* max size of auxiliary scalars */ 
    max_nev,            /* max number of converged pairs */
    own_vecs,           /* number of global vecs */
    own_scalars;        /* number of local scalars */
  Vec *free_vecs;       /* free global vectors */
  PetscScalar
    *free_scalars;      /* free scalars */
  PetscInt state;       /* method states:
                            0: preconfiguring
                            1: configuring
                            2: running
                        */
} dvdBlackboard;

#define DVD_STATE_PRECONF 0
#define DVD_STATE_CONF 1
#define DVD_STATE_RUN 2

/* Shared types */
typedef void* dvdPrecondData; // DEPRECATED!!
typedef PetscErrorCode (*dvdPrecond)(dvdDashboard*, PetscInt i, Vec x, Vec Px);
typedef PetscInt (*dvdCallback)(dvdDashboard*);
typedef PetscInt (*e_Vchanged_type)(dvdDashboard*, PetscInt s_imm,
                         PetscInt e_imm, PetscInt s_new, PetscInt e_new);
typedef PetscTruth (*isRestarting_type)(dvdDashboard*);
typedef PetscInt (*e_newIteration_type)(dvdDashboard*);
typedef PetscInt (*improveX_type)(dvdDashboard*, Vec *D, PetscInt max_size_D,
                                  PetscInt r_s, PetscInt r_e, PetscInt *size_D);

/* Structures for blas */
typedef PetscErrorCode (*DvdReductionPostF)(PetscScalar*,PetscInt,void*);
typedef struct {
  PetscScalar
    *out;               /* final vector */
  PetscInt
    size_out;           /* size of out */
  DvdReductionPostF
    f;                  /* function called after the reduction */
  void *ptr;
} DvdReductionChunk;  

typedef struct {
  PetscScalar
    *in,                /* vector to sum-up with more nodes */
    *out;               /* final vector */
  PetscInt size_in,     /* size of in */
    max_size_in;        /* max size of in */
  DvdReductionChunk
    *ops;               /* vector of reduction operations */
  PetscInt
    size_ops,           /* size of ops */
    max_size_ops;       /* max size of ops */
  MPI_Comm comm;        /* MPI communicator */
} DvdReduction;

typedef struct {
  PetscInt i0, i1, i2, ld, s0, e0, s1, e1;
  PetscScalar *M;
} DvdMult_copy_func;

/* Routines for initV step */
PetscInt dvd_initV_classic(dvdDashboard *d, dvdBlackboard *b, PetscInt k);
PetscInt dvd_initV_user(dvdDashboard *d, dvdBlackboard *b, Vec *userV,
                        PetscInt size_userV, PetscInt k);
PetscInt dvd_initV_krylov(dvdDashboard *d, dvdBlackboard *b, PetscInt k);

/* Routines for calcPairs step */
PetscInt dvd_calcpairs_rr(dvdDashboard *d, dvdBlackboard *b);
PetscInt dvd_calcpairs_qz(dvdDashboard *d, dvdBlackboard *b, IP ip);

/* Routines for improveX step */
PetscInt dvd_improvex_jd(dvdDashboard *d, dvdBlackboard *b, KSP ksp,
                         PetscInt max_bs);
PetscInt dvd_improvex_jd_proj_uv(dvdDashboard *d, dvdBlackboard *b,
                                 ProjType_t p);
PetscInt dvd_improvex_jd_lit_const(dvdDashboard *d, dvdBlackboard *b,
                                   PetscInt maxits, PetscReal tol,
                                   PetscReal fix);

/* Routines for testConv step */
PetscInt dvd_testconv_basic(dvdDashboard *d, dvdBlackboard *b);
PetscInt dvd_testconv_slepc(dvdDashboard *d, dvdBlackboard *b);

/* Routines for management of V */
PetscInt dvd_managementV_basic(dvdDashboard *d, dvdBlackboard *b,
                               PetscInt bs, PetscInt max_size_V,
                               PetscInt restart_size_X, PetscInt plusk,
                               PetscTruth harm);

/* Some utilities */
PetscErrorCode dvd_static_precond_PC(dvdDashboard *d, dvdBlackboard *b, PC pc);
PetscErrorCode dvd_jacobi_precond(dvdDashboard *d, dvdBlackboard *b);
PetscErrorCode dvd_profiler(dvdDashboard *d, dvdBlackboard *b);
PetscErrorCode dvd_prof_init();
PetscErrorCode dvd_harm_conf(dvdDashboard *d, dvdBlackboard *b,
                             HarmType_t mode, PetscTruth fixedTarget,
                             PetscScalar t);

/* Methods */
PetscErrorCode dvd_schm_basic_preconf(dvdDashboard *d, dvdBlackboard *b,
  PetscInt max_size_V, PetscInt min_size_V, PetscInt bs, PetscInt ini_size_V,
  Vec *initV, PetscInt size_initV, PetscInt plusk, PC pc, HarmType_t harmMode,
  KSP ksp, InitType_t init);
PetscErrorCode dvd_schm_basic_conf(dvdDashboard *d, dvdBlackboard *b,
  PetscInt max_size_V, PetscInt min_size_V, PetscInt bs, PetscInt ini_size_V,
  Vec *initV, PetscInt size_initV, PetscInt plusk, PC pc, IP ip,
  HarmType_t harmMode, PetscTruth fixedTarget, PetscScalar t, KSP ksp,
  PetscReal fix, InitType_t init);

/* BLAS routines */
PetscErrorCode dvd_blas_prof_init();
PetscErrorCode SlepcDenseMatProd(PetscScalar *C, PetscInt _ldC, PetscScalar b,
  PetscScalar a,
  const PetscScalar *A, PetscInt _ldA, PetscInt rA, PetscInt cA, PetscTruth At,
  const PetscScalar *B, PetscInt _ldB, PetscInt rB, PetscInt cB, PetscTruth Bt);
PetscErrorCode SlepcDenseMatProdTriang(
  PetscScalar *C, MatType_t sC, PetscInt ldC,
  const PetscScalar *A, MatType_t sA, PetscInt ldA, PetscInt rA, PetscInt cA,
  PetscTruth At,
  const PetscScalar *B, MatType_t sB, PetscInt ldB, PetscInt rB, PetscInt cB,
  PetscTruth Bt);
PetscErrorCode SlepcDenseMatInvProd(
  PetscScalar *A, PetscInt _ldA, PetscInt dimA,
  PetscScalar *B, PetscInt _ldB, PetscInt rB, PetscInt cB, PetscInt *auxI);
PetscErrorCode SlepcDenseNorm(PetscScalar *A, PetscInt ldA, PetscInt _rA,
                              PetscInt cA, PetscScalar *eigi);
PetscErrorCode SlepcDenseOrth(PetscScalar *A, PetscInt _ldA, PetscInt _rA,
                              PetscInt _cA, PetscScalar *auxS, PetscInt _lauxS,
                              PetscInt *ncA);
PetscErrorCode SlepcDenseCopy(PetscScalar *Y, PetscInt ldY, PetscScalar *X,
                              PetscInt ldX, PetscInt rX, PetscInt cX);
PetscErrorCode SlepcDenseCopyTriang(PetscScalar *Y, MatType_t sY, PetscInt ldY,
                                    PetscScalar *X, MatType_t sX, PetscInt ldX,
                                    PetscInt rX, PetscInt cX);
PetscErrorCode SlepcUpdateVectorsZ(Vec *Y, PetscScalar beta, PetscScalar alpha,
  Vec *X, PetscInt cX, const PetscScalar *M, PetscInt ldM, PetscInt rM,
  PetscInt cM);
PetscErrorCode SlepcUpdateVectorsS(Vec *Y, PetscInt dY, PetscScalar beta,
  PetscScalar alpha, Vec *X, PetscInt cX, PetscInt dX, const PetscScalar *M,
  PetscInt ldM, PetscInt rM, PetscInt cM);
PetscErrorCode SlepcUpdateVectorsD(Vec *X, PetscInt cX, PetscScalar alpha,
  const PetscScalar *M, PetscInt ldM, PetscInt rM, PetscInt cM,
  PetscScalar *work, PetscInt lwork);
PetscErrorCode VecsMult(PetscScalar *M, MatType_t sM, PetscInt ldM,
                        Vec *U, PetscInt sU, PetscInt eU,
                        Vec *V, PetscInt sV, PetscInt eV,
                        PetscScalar *workS0, PetscScalar *workS1);
PetscErrorCode VecsMultS(PetscScalar *M, MatType_t sM, PetscInt ldM,
                         Vec *U, PetscInt sU, PetscInt eU,
                         Vec *V, PetscInt sV, PetscInt eV, DvdReduction *r,
                         DvdMult_copy_func *sr);
PetscErrorCode VecsMultIc(PetscScalar *M, MatType_t sM, PetscInt ldM,
                          PetscInt rM, PetscInt cM, Vec V);
PetscErrorCode VecsMultIb(PetscScalar *M, MatType_t sM, PetscInt ldM,
                          PetscInt rM, PetscInt cM, PetscScalar *auxS,
                          Vec V);
PetscErrorCode VecsMultIa(PetscScalar *M, MatType_t sM, PetscInt ldM,
                          Vec *U, PetscInt sU, PetscInt eU,
                          Vec *V, PetscInt sV, PetscInt eV);
PetscErrorCode SlepcAllReduceSumBegin(DvdReductionChunk *ops,
                                      PetscInt max_size_ops,
                                      PetscScalar *in, PetscScalar *out,
                                      PetscInt max_size_in, DvdReduction *r,
                                      MPI_Comm comm);
PetscErrorCode SlepcAllReduceSum(DvdReduction *r, PetscInt size_in,
                                 DvdReductionPostF f, void *ptr,
                                 PetscScalar **in);
PetscErrorCode SlepcAllReduceSumEnd(DvdReduction *r);
PetscErrorCode dvd_orthV(IP ip, Vec *DS, PetscInt size_DS, Vec *cX,
                         PetscInt size_cX, Vec *V, PetscInt V_new_s,
                         PetscInt V_new_e, PetscScalar *auxS, Vec auxV,
                         PetscRandom rand);
PetscErrorCode dvd_compute_eigenvectors(PetscInt n_, PetscScalar *S,
  PetscInt ldS_, PetscScalar *T, PetscInt ldT_, PetscScalar *pX,
  PetscInt ldpX_, PetscScalar *pY, PetscInt ldpY_, PetscScalar *auxS,
  PetscInt size_auxS, PetscTruth doProd);

/* SLEPc interface routines */
PetscErrorCode SLEPcNotImplemented();
PetscErrorCode EPSCreate_DAVIDSON(EPS eps);
PetscErrorCode EPSDestroy_DAVIDSON(EPS eps);
PetscErrorCode EPSSetUp_DAVIDSON(EPS eps);
PetscErrorCode EPSSolve_DAVIDSON(EPS eps);
PetscErrorCode EPSComputeVectors_QZ(EPS eps);
PetscErrorCode EPSDAVIDSONSetKrylovStart_DAVIDSON(EPS eps,PetscTruth krylovstart);
PetscErrorCode EPSDAVIDSONGetKrylovStart_DAVIDSON(EPS eps,PetscTruth *krylovstart);
PetscErrorCode EPSDAVIDSONSetBlockSize_DAVIDSON(EPS eps,PetscInt blocksize);
PetscErrorCode EPSDAVIDSONGetBlockSize_DAVIDSON(EPS eps,PetscInt *blocksize);
PetscErrorCode EPSDAVIDSONSetRestart_DAVIDSON(EPS eps,PetscInt minv,PetscInt plusk);
PetscErrorCode EPSDAVIDSONGetRestart_DAVIDSON(EPS eps,PetscInt *minv,PetscInt *plusk);
PetscErrorCode EPSDAVIDSONGetInitialSize_DAVIDSON(EPS eps,PetscInt *initialsize);
PetscErrorCode EPSDAVIDSONSetInitialSize_DAVIDSON(EPS eps,PetscInt initialsize);
PetscErrorCode EPSDAVIDSONGetFix_DAVIDSON(EPS eps,PetscReal *fix);
PetscErrorCode EPSDAVIDSONSetFix_DAVIDSON(EPS eps,PetscReal fix);

typedef struct {
  /**** Solver options ****/
  PetscInt blocksize,     /* block size */
    initialsize,          /* initial size of V */
    minv,                 /* size of V after restarting */
    plusk;                /* keep plusk eigenvectors from the last iteration */
  PetscTruth ipB;         /* truth if V'B*V=I */
  PetscInt   method;      /* method for improving the approximate solution */
  PetscReal  fix;         /* the fix parameter */
  PetscTruth krylovstart; /* true if the starting subspace is a Krylov basis */

  /**** Solver data ****/
  dvdDashboard ddb;

  /**** Things to destroy ****/
  PetscScalar *wS;
  Vec         *wV;
  PetscInt    size_wV;
} EPS_DAVIDSON;

