/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/
/*
   BV (basis vectors) interface routines, callable by users
*/

#include <slepc/private/bvimpl.h>            /*I "slepcbv.h" I*/

PetscClassId     BV_CLASSID = 0;
PetscLogEvent    BV_Create = 0,BV_Copy = 0,BV_Mult = 0,BV_MultVec = 0,BV_MultInPlace = 0,BV_Dot = 0,BV_DotVec = 0,BV_Orthogonalize = 0,BV_OrthogonalizeVec = 0,BV_Scale = 0,BV_Norm = 0,BV_NormVec = 0,BV_Normalize = 0,BV_SetRandom = 0,BV_MatMult = 0,BV_MatMultVec = 0,BV_MatProject = 0,BV_SVDAndRank = 0;
static PetscBool BVPackageInitialized = PETSC_FALSE;
MPI_Op MPIU_TSQR = 0,MPIU_LAPY2;

const char *BVOrthogTypes[] = {"CGS","MGS","BVOrthogType","BV_ORTHOG_",0};
const char *BVOrthogRefineTypes[] = {"IFNEEDED","NEVER","ALWAYS","BVOrthogRefineType","BV_ORTHOG_REFINE_",0};
const char *BVOrthogBlockTypes[] = {"GS","CHOL","TSQR","TSQRCHOL","SVQB","BVOrthogBlockType","BV_ORTHOG_BLOCK_",0};
const char *BVMatMultTypes[] = {"VECS","MAT","MAT_SAVE","BVMatMultType","BV_MATMULT_",0};
const char *BVSVDMethods[] = {"REFINE","QR","QR_CAA","BVSVDMethod","BV_SVD_METHOD_",0};

/*@C
   BVFinalizePackage - This function destroys everything in the Slepc interface
   to the BV package. It is called from SlepcFinalize().

   Level: developer

.seealso: SlepcFinalize()
@*/
PetscErrorCode BVFinalizePackage(void)
{
  PetscFunctionBegin;
  CHKERRQ(PetscFunctionListDestroy(&BVList));
  CHKERRMPI(MPI_Op_free(&MPIU_TSQR));
  CHKERRMPI(MPI_Op_free(&MPIU_LAPY2));
  BVPackageInitialized = PETSC_FALSE;
  BVRegisterAllCalled  = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/*@C
   BVInitializePackage - This function initializes everything in the BV package.
   It is called from PetscDLLibraryRegister() when using dynamic libraries, and
   on the first call to BVCreate() when using static libraries.

   Level: developer

.seealso: SlepcInitialize()
@*/
PetscErrorCode BVInitializePackage(void)
{
  char           logList[256];
  PetscBool      opt,pkg;
  PetscClassId   classids[1];

  PetscFunctionBegin;
  if (BVPackageInitialized) PetscFunctionReturn(0);
  BVPackageInitialized = PETSC_TRUE;
  /* Register Classes */
  CHKERRQ(PetscClassIdRegister("Basis Vectors",&BV_CLASSID));
  /* Register Constructors */
  CHKERRQ(BVRegisterAll());
  /* Register Events */
  CHKERRQ(PetscLogEventRegister("BVCreate",BV_CLASSID,&BV_Create));
  CHKERRQ(PetscLogEventRegister("BVCopy",BV_CLASSID,&BV_Copy));
  CHKERRQ(PetscLogEventRegister("BVMult",BV_CLASSID,&BV_Mult));
  CHKERRQ(PetscLogEventRegister("BVMultVec",BV_CLASSID,&BV_MultVec));
  CHKERRQ(PetscLogEventRegister("BVMultInPlace",BV_CLASSID,&BV_MultInPlace));
  CHKERRQ(PetscLogEventRegister("BVDot",BV_CLASSID,&BV_Dot));
  CHKERRQ(PetscLogEventRegister("BVDotVec",BV_CLASSID,&BV_DotVec));
  CHKERRQ(PetscLogEventRegister("BVOrthogonalize",BV_CLASSID,&BV_Orthogonalize));
  CHKERRQ(PetscLogEventRegister("BVOrthogonalizeV",BV_CLASSID,&BV_OrthogonalizeVec));
  CHKERRQ(PetscLogEventRegister("BVScale",BV_CLASSID,&BV_Scale));
  CHKERRQ(PetscLogEventRegister("BVNorm",BV_CLASSID,&BV_Norm));
  CHKERRQ(PetscLogEventRegister("BVNormVec",BV_CLASSID,&BV_NormVec));
  CHKERRQ(PetscLogEventRegister("BVNormalize",BV_CLASSID,&BV_Normalize));
  CHKERRQ(PetscLogEventRegister("BVSetRandom",BV_CLASSID,&BV_SetRandom));
  CHKERRQ(PetscLogEventRegister("BVMatMult",BV_CLASSID,&BV_MatMult));
  CHKERRQ(PetscLogEventRegister("BVMatMultVec",BV_CLASSID,&BV_MatMultVec));
  CHKERRQ(PetscLogEventRegister("BVMatProject",BV_CLASSID,&BV_MatProject));
  CHKERRQ(PetscLogEventRegister("BVSVDAndRank",BV_CLASSID,&BV_SVDAndRank));
  /* MPI reduction operation used in BVOrthogonalize */
  CHKERRMPI(MPI_Op_create(SlepcGivensPacked,PETSC_FALSE,&MPIU_TSQR));
  CHKERRMPI(MPI_Op_create(SlepcPythag,PETSC_TRUE,&MPIU_LAPY2));
  /* Process Info */
  classids[0] = BV_CLASSID;
  CHKERRQ(PetscInfoProcessClass("bv",1,&classids[0]));
  /* Process summary exclusions */
  CHKERRQ(PetscOptionsGetString(NULL,NULL,"-log_exclude",logList,sizeof(logList),&opt));
  if (opt) {
    CHKERRQ(PetscStrInList("bv",logList,',',&pkg));
    if (pkg) CHKERRQ(PetscLogEventDeactivateClass(BV_CLASSID));
  }
  /* Register package finalizer */
  CHKERRQ(PetscRegisterFinalize(BVFinalizePackage));
  PetscFunctionReturn(0);
}

/*@C
   BVDestroy - Destroys BV context that was created with BVCreate().

   Collective on bv

   Input Parameter:
.  bv - the basis vectors context

   Level: beginner

.seealso: BVCreate()
@*/
PetscErrorCode BVDestroy(BV *bv)
{
  PetscFunctionBegin;
  if (!*bv) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(*bv,BV_CLASSID,1);
  PetscCheck(!(*bv)->lsplit,PetscObjectComm((PetscObject)(*bv)),PETSC_ERR_ARG_WRONGSTATE,"Must call BVRestoreSplit before destroying the BV");
  if (--((PetscObject)(*bv))->refct > 0) { *bv = 0; PetscFunctionReturn(0); }
  if ((*bv)->ops->destroy) CHKERRQ((*(*bv)->ops->destroy)(*bv));
  CHKERRQ(VecDestroy(&(*bv)->t));
  CHKERRQ(MatDestroy(&(*bv)->matrix));
  CHKERRQ(VecDestroy(&(*bv)->Bx));
  CHKERRQ(VecDestroy(&(*bv)->buffer));
  CHKERRQ(BVDestroy(&(*bv)->cached));
  CHKERRQ(BVDestroy(&(*bv)->L));
  CHKERRQ(BVDestroy(&(*bv)->R));
  CHKERRQ(PetscFree((*bv)->work));
  CHKERRQ(PetscFree2((*bv)->h,(*bv)->c));
  CHKERRQ(VecDestroy(&(*bv)->omega));
  CHKERRQ(MatDestroy(&(*bv)->Acreate));
  CHKERRQ(MatDestroy(&(*bv)->Aget));
  CHKERRQ(MatDestroy(&(*bv)->Abuffer));
  CHKERRQ(PetscRandomDestroy(&(*bv)->rand));
  CHKERRQ(PetscHeaderDestroy(bv));
  PetscFunctionReturn(0);
}

/*@
   BVCreate - Creates a basis vectors context.

   Collective

   Input Parameter:
.  comm - MPI communicator

   Output Parameter:
.  newbv - location to put the basis vectors context

   Level: beginner

.seealso: BVSetUp(), BVDestroy(), BV
@*/
PetscErrorCode BVCreate(MPI_Comm comm,BV *newbv)
{
  BV             bv;

  PetscFunctionBegin;
  PetscValidPointer(newbv,2);
  *newbv = 0;
  CHKERRQ(BVInitializePackage());
  CHKERRQ(SlepcHeaderCreate(bv,BV_CLASSID,"BV","Basis Vectors","BV",comm,BVDestroy,BVView));

  bv->t            = NULL;
  bv->n            = -1;
  bv->N            = -1;
  bv->m            = 0;
  bv->l            = 0;
  bv->k            = 0;
  bv->nc           = 0;
  bv->orthog_type  = BV_ORTHOG_CGS;
  bv->orthog_ref   = BV_ORTHOG_REFINE_IFNEEDED;
  bv->orthog_eta   = 0.7071;
  bv->orthog_block = BV_ORTHOG_BLOCK_GS;
  bv->matrix       = NULL;
  bv->indef        = PETSC_FALSE;
  bv->vmm          = BV_MATMULT_MAT;
  bv->rrandom      = PETSC_FALSE;
  bv->deftol       = 10*PETSC_MACHINE_EPSILON;

  bv->Bx           = NULL;
  bv->buffer       = NULL;
  bv->Abuffer      = NULL;
  bv->xid          = 0;
  bv->xstate       = 0;
  bv->cv[0]        = NULL;
  bv->cv[1]        = NULL;
  bv->ci[0]        = -1;
  bv->ci[1]        = -1;
  bv->st[0]        = -1;
  bv->st[1]        = -1;
  bv->id[0]        = 0;
  bv->id[1]        = 0;
  bv->h            = NULL;
  bv->c            = NULL;
  bv->omega        = NULL;
  bv->defersfo     = PETSC_FALSE;
  bv->cached       = NULL;
  bv->bvstate      = 0;
  bv->L            = NULL;
  bv->R            = NULL;
  bv->lstate       = 0;
  bv->rstate       = 0;
  bv->lsplit       = 0;
  bv->issplit      = 0;
  bv->splitparent  = NULL;
  bv->rand         = NULL;
  bv->rrandom      = PETSC_FALSE;
  bv->Acreate      = NULL;
  bv->Aget         = NULL;
  bv->cuda         = PETSC_FALSE;
  bv->sfocalled    = PETSC_FALSE;
  bv->work         = NULL;
  bv->lwork        = 0;
  bv->data         = NULL;

  *newbv = bv;
  PetscFunctionReturn(0);
}

/*@
   BVCreateFromMat - Creates a basis vectors object from a dense Mat object.

   Collective on A

   Input Parameter:
.  A - a dense tall-skinny matrix

   Output Parameter:
.  bv - the new basis vectors context

   Notes:
   The matrix values are copied to the BV data storage, memory is not shared.

   The communicator of the BV object will be the same as A, and so will be
   the dimensions.

   Level: intermediate

.seealso: BVCreate(), BVDestroy(), BVCreateMat()
@*/
PetscErrorCode BVCreateFromMat(Mat A,BV *bv)
{
  PetscInt       n,N,k;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscCheckTypeNames(A,MATSEQDENSE,MATMPIDENSE);

  CHKERRQ(MatGetSize(A,&N,&k));
  CHKERRQ(MatGetLocalSize(A,&n,NULL));
  CHKERRQ(BVCreate(PetscObjectComm((PetscObject)A),bv));
  CHKERRQ(BVSetSizes(*bv,n,N,k));

  (*bv)->Acreate = A;
  CHKERRQ(PetscObjectReference((PetscObject)A));
  PetscFunctionReturn(0);
}

/*@
   BVInsertVec - Insert a vector into the specified column.

   Collective on V

   Input Parameters:
+  V - basis vectors
.  j - the column of V to be overwritten
-  w - the vector to be copied

   Level: intermediate

.seealso: BVInsertVecs()
@*/
PetscErrorCode BVInsertVec(BV V,PetscInt j,Vec w)
{
  PetscInt       n,N;
  Vec            v;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(V,BV_CLASSID,1);
  PetscValidLogicalCollectiveInt(V,j,2);
  PetscValidHeaderSpecific(w,VEC_CLASSID,3);
  PetscValidType(V,1);
  BVCheckSizes(V,1);
  PetscCheckSameComm(V,1,w,3);

  CHKERRQ(VecGetSize(w,&N));
  CHKERRQ(VecGetLocalSize(w,&n));
  PetscCheck(N==V->N && n==V->n,PetscObjectComm((PetscObject)V),PETSC_ERR_ARG_INCOMP,"Vec sizes (global %" PetscInt_FMT ", local %" PetscInt_FMT ") do not match BV sizes (global %" PetscInt_FMT ", local %" PetscInt_FMT ")",N,n,V->N,V->n);
  PetscCheck(j>=-V->nc && j<V->m,PetscObjectComm((PetscObject)V),PETSC_ERR_ARG_OUTOFRANGE,"Argument j has wrong value %" PetscInt_FMT ", should be between %" PetscInt_FMT " and %" PetscInt_FMT,j,-V->nc,V->m-1);

  CHKERRQ(BVGetColumn(V,j,&v));
  CHKERRQ(VecCopy(w,v));
  CHKERRQ(BVRestoreColumn(V,j,&v));
  CHKERRQ(PetscObjectStateIncrease((PetscObject)V));
  PetscFunctionReturn(0);
}

/*@
   BVInsertVecs - Insert a set of vectors into the specified columns.

   Collective on V

   Input Parameters:
+  V - basis vectors
.  s - first column of V to be overwritten
.  W - set of vectors to be copied
-  orth - flag indicating if the vectors must be orthogonalized

   Input/Output Parameter:
.  m - number of input vectors, on output the number of linearly independent
       vectors

   Notes:
   Copies the contents of vectors W to V(:,s:s+n). If the orthogonalization
   flag is set, then the vectors are copied one by one and then orthogonalized
   against the previous ones. If any of them is linearly dependent then it
   is discarded and the value of m is decreased.

   Level: intermediate

.seealso: BVInsertVec(), BVOrthogonalizeColumn()
@*/
PetscErrorCode BVInsertVecs(BV V,PetscInt s,PetscInt *m,Vec *W,PetscBool orth)
{
  PetscInt       n,N,i,ndep;
  PetscBool      lindep;
  PetscReal      norm;
  Vec            v;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(V,BV_CLASSID,1);
  PetscValidLogicalCollectiveInt(V,s,2);
  PetscValidIntPointer(m,3);
  PetscValidLogicalCollectiveInt(V,*m,3);
  if (!*m) PetscFunctionReturn(0);
  PetscCheck(*m>0,PetscObjectComm((PetscObject)V),PETSC_ERR_ARG_OUTOFRANGE,"Number of vectors (given %" PetscInt_FMT ") cannot be negative",*m);
  PetscValidPointer(W,4);
  PetscValidHeaderSpecific(*W,VEC_CLASSID,4);
  PetscValidLogicalCollectiveBool(V,orth,5);
  PetscValidType(V,1);
  BVCheckSizes(V,1);
  PetscCheckSameComm(V,1,*W,4);

  CHKERRQ(VecGetSize(*W,&N));
  CHKERRQ(VecGetLocalSize(*W,&n));
  PetscCheck(N==V->N && n==V->n,PetscObjectComm((PetscObject)V),PETSC_ERR_ARG_INCOMP,"Vec sizes (global %" PetscInt_FMT ", local %" PetscInt_FMT ") do not match BV sizes (global %" PetscInt_FMT ", local %" PetscInt_FMT ")",N,n,V->N,V->n);
  PetscCheck(s>=0 && s<V->m,PetscObjectComm((PetscObject)V),PETSC_ERR_ARG_OUTOFRANGE,"Argument s has wrong value %" PetscInt_FMT ", should be between 0 and %" PetscInt_FMT,s,V->m-1);
  PetscCheck(s+(*m)<=V->m,PetscObjectComm((PetscObject)V),PETSC_ERR_ARG_OUTOFRANGE,"Too many vectors provided, there is only room for %" PetscInt_FMT,V->m);

  ndep = 0;
  for (i=0;i<*m;i++) {
    CHKERRQ(BVGetColumn(V,s+i-ndep,&v));
    CHKERRQ(VecCopy(W[i],v));
    CHKERRQ(BVRestoreColumn(V,s+i-ndep,&v));
    if (orth) {
      CHKERRQ(BVOrthogonalizeColumn(V,s+i-ndep,NULL,&norm,&lindep));
      if (norm==0.0 || lindep) {
        CHKERRQ(PetscInfo(V,"Removing linearly dependent vector %" PetscInt_FMT "\n",i));
        ndep++;
      } else {
        CHKERRQ(BVScaleColumn(V,s+i-ndep,1.0/norm));
      }
    }
  }
  *m -= ndep;
  CHKERRQ(PetscObjectStateIncrease((PetscObject)V));
  PetscFunctionReturn(0);
}

/*@
   BVInsertConstraints - Insert a set of vectors as constraints.

   Collective on V

   Input Parameters:
+  V - basis vectors
-  C - set of vectors to be inserted as constraints

   Input/Output Parameter:
.  nc - number of input vectors, on output the number of linearly independent
       vectors

   Notes:
   The constraints are relevant only during orthogonalization. Constraint
   vectors span a subspace that is deflated in every orthogonalization
   operation, so they are intended for removing those directions from the
   orthogonal basis computed in regular BV columns.

   Constraints are not stored in regular BV columns, but in a special part of
   the storage. They can be accessed with negative indices in BVGetColumn().

   This operation is DESTRUCTIVE, meaning that all data contained in the
   columns of V is lost. This is typically invoked just after creating the BV.
   Once a set of constraints has been set, it is not allowed to call this
   function again.

   The vectors are copied one by one and then orthogonalized against the
   previous ones. If any of them is linearly dependent then it is discarded
   and the value of nc is decreased. The behaviour is similar to BVInsertVecs().

   Level: advanced

.seealso: BVInsertVecs(), BVOrthogonalizeColumn(), BVGetColumn(), BVGetNumConstraints()
@*/
PetscErrorCode BVInsertConstraints(BV V,PetscInt *nc,Vec *C)
{
  PetscInt       msave;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(V,BV_CLASSID,1);
  PetscValidIntPointer(nc,2);
  PetscValidLogicalCollectiveInt(V,*nc,2);
  if (!*nc) PetscFunctionReturn(0);
  PetscCheck(*nc>0,PetscObjectComm((PetscObject)V),PETSC_ERR_ARG_OUTOFRANGE,"Number of constraints (given %" PetscInt_FMT ") cannot be negative",*nc);
  PetscValidPointer(C,3);
  PetscValidHeaderSpecific(*C,VEC_CLASSID,3);
  PetscValidType(V,1);
  BVCheckSizes(V,1);
  PetscCheckSameComm(V,1,*C,3);
  PetscCheck(!V->issplit,PetscObjectComm((PetscObject)V),PETSC_ERR_ARG_WRONGSTATE,"Operation not permitted for a BV obtained from BVGetSplit");
  PetscCheck(!V->nc,PetscObjectComm((PetscObject)V),PETSC_ERR_ARG_WRONGSTATE,"Constraints already present in this BV object");
  PetscCheck(V->ci[0]==-1 && V->ci[1]==-1,PetscObjectComm((PetscObject)V),PETSC_ERR_SUP,"Cannot call BVInsertConstraints after BVGetColumn");

  msave = V->m;
  CHKERRQ(BVResize(V,*nc+V->m,PETSC_FALSE));
  CHKERRQ(BVInsertVecs(V,0,nc,C,PETSC_TRUE));
  V->nc = *nc;
  V->m  = msave;
  V->ci[0] = -V->nc-1;
  V->ci[1] = -V->nc-1;
  CHKERRQ(PetscObjectStateIncrease((PetscObject)V));
  PetscFunctionReturn(0);
}

/*@C
   BVSetOptionsPrefix - Sets the prefix used for searching for all
   BV options in the database.

   Logically Collective on bv

   Input Parameters:
+  bv     - the basis vectors context
-  prefix - the prefix string to prepend to all BV option requests

   Notes:
   A hyphen (-) must NOT be given at the beginning of the prefix name.
   The first character of all runtime options is AUTOMATICALLY the
   hyphen.

   Level: advanced

.seealso: BVAppendOptionsPrefix(), BVGetOptionsPrefix()
@*/
PetscErrorCode BVSetOptionsPrefix(BV bv,const char *prefix)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(bv,BV_CLASSID,1);
  CHKERRQ(PetscObjectSetOptionsPrefix((PetscObject)bv,prefix));
  PetscFunctionReturn(0);
}

/*@C
   BVAppendOptionsPrefix - Appends to the prefix used for searching for all
   BV options in the database.

   Logically Collective on bv

   Input Parameters:
+  bv     - the basis vectors context
-  prefix - the prefix string to prepend to all BV option requests

   Notes:
   A hyphen (-) must NOT be given at the beginning of the prefix name.
   The first character of all runtime options is AUTOMATICALLY the
   hyphen.

   Level: advanced

.seealso: BVSetOptionsPrefix(), BVGetOptionsPrefix()
@*/
PetscErrorCode BVAppendOptionsPrefix(BV bv,const char *prefix)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(bv,BV_CLASSID,1);
  CHKERRQ(PetscObjectAppendOptionsPrefix((PetscObject)bv,prefix));
  PetscFunctionReturn(0);
}

/*@C
   BVGetOptionsPrefix - Gets the prefix used for searching for all
   BV options in the database.

   Not Collective

   Input Parameters:
.  bv - the basis vectors context

   Output Parameters:
.  prefix - pointer to the prefix string used, is returned

   Note:
   On the Fortran side, the user should pass in a string 'prefix' of
   sufficient length to hold the prefix.

   Level: advanced

.seealso: BVSetOptionsPrefix(), BVAppendOptionsPrefix()
@*/
PetscErrorCode BVGetOptionsPrefix(BV bv,const char *prefix[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(bv,BV_CLASSID,1);
  PetscValidPointer(prefix,2);
  CHKERRQ(PetscObjectGetOptionsPrefix((PetscObject)bv,prefix));
  PetscFunctionReturn(0);
}

static PetscErrorCode BVView_Default(BV bv,PetscViewer viewer)
{
  PetscInt          j;
  Vec               v;
  PetscViewerFormat format;
  PetscBool         isascii,ismatlab=PETSC_FALSE;
  const char        *bvname,*name;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii));
  if (isascii) {
    CHKERRQ(PetscViewerGetFormat(viewer,&format));
    if (format == PETSC_VIEWER_ASCII_MATLAB) ismatlab = PETSC_TRUE;
  }
  if (ismatlab) {
    CHKERRQ(PetscObjectGetName((PetscObject)bv,&bvname));
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"%s=[];\n",bvname));
  }
  for (j=-bv->nc;j<bv->m;j++) {
    CHKERRQ(BVGetColumn(bv,j,&v));
    CHKERRQ(VecView(v,viewer));
    if (ismatlab) {
      CHKERRQ(PetscObjectGetName((PetscObject)v,&name));
      CHKERRQ(PetscViewerASCIIPrintf(viewer,"%s=[%s,%s];clear %s\n",bvname,bvname,name,name));
    }
    CHKERRQ(BVRestoreColumn(bv,j,&v));
  }
  PetscFunctionReturn(0);
}

/*@C
   BVView - Prints the BV data structure.

   Collective on bv

   Input Parameters:
+  bv     - the BV context
-  viewer - optional visualization context

   Note:
   The available visualization contexts include
+     PETSC_VIEWER_STDOUT_SELF - standard output (default)
-     PETSC_VIEWER_STDOUT_WORLD - synchronized standard
         output where only the first processor opens
         the file.  All other processors send their
         data to the first processor to print.

   The user can open an alternative visualization contexts with
   PetscViewerASCIIOpen() (output to a specified file).

   Level: beginner

.seealso: BVCreate()
@*/
PetscErrorCode BVView(BV bv,PetscViewer viewer)
{
  PetscBool         isascii;
  PetscViewerFormat format;
  const char        *orthname[2] = {"classical","modified"};
  const char        *refname[3] = {"if needed","never","always"};

  PetscFunctionBegin;
  PetscValidHeaderSpecific(bv,BV_CLASSID,1);
  if (!viewer) {
    CHKERRQ(PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)bv),&viewer));
  }
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);

  CHKERRQ(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii));
  if (isascii) {
    CHKERRQ(PetscObjectPrintClassNamePrefixType((PetscObject)bv,viewer));
    CHKERRQ(PetscViewerGetFormat(viewer,&format));
    if (format == PETSC_VIEWER_ASCII_INFO || format == PETSC_VIEWER_ASCII_INFO_DETAIL) {
      CHKERRQ(PetscViewerASCIIPrintf(viewer,"  %" PetscInt_FMT " columns of global length %" PetscInt_FMT "%s\n",bv->m,bv->N,bv->cuda?" (CUDA)":""));
      if (bv->nc>0) {
        CHKERRQ(PetscViewerASCIIPrintf(viewer,"  number of constraints: %" PetscInt_FMT "\n",bv->nc));
      }
      CHKERRQ(PetscViewerASCIIPrintf(viewer,"  vector orthogonalization method: %s Gram-Schmidt\n",orthname[bv->orthog_type]));
      switch (bv->orthog_ref) {
        case BV_ORTHOG_REFINE_IFNEEDED:
          CHKERRQ(PetscViewerASCIIPrintf(viewer,"  orthogonalization refinement: %s (eta: %g)\n",refname[bv->orthog_ref],(double)bv->orthog_eta));
          break;
        case BV_ORTHOG_REFINE_NEVER:
        case BV_ORTHOG_REFINE_ALWAYS:
          CHKERRQ(PetscViewerASCIIPrintf(viewer,"  orthogonalization refinement: %s\n",refname[bv->orthog_ref]));
          break;
      }
      CHKERRQ(PetscViewerASCIIPrintf(viewer,"  block orthogonalization method: %s\n",BVOrthogBlockTypes[bv->orthog_block]));
      if (bv->matrix) {
        if (bv->indef) {
          CHKERRQ(PetscViewerASCIIPrintf(viewer,"  indefinite inner product\n"));
        } else {
          CHKERRQ(PetscViewerASCIIPrintf(viewer,"  non-standard inner product\n"));
        }
        CHKERRQ(PetscViewerASCIIPrintf(viewer,"  tolerance for definite inner product: %g\n",(double)bv->deftol));
        CHKERRQ(PetscViewerASCIIPrintf(viewer,"  inner product matrix:\n"));
        CHKERRQ(PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_INFO));
        CHKERRQ(PetscViewerASCIIPushTab(viewer));
        CHKERRQ(MatView(bv->matrix,viewer));
        CHKERRQ(PetscViewerASCIIPopTab(viewer));
        CHKERRQ(PetscViewerPopFormat(viewer));
      }
      switch (bv->vmm) {
        case BV_MATMULT_VECS:
          CHKERRQ(PetscViewerASCIIPrintf(viewer,"  doing matmult as matrix-vector products\n"));
          break;
        case BV_MATMULT_MAT:
          CHKERRQ(PetscViewerASCIIPrintf(viewer,"  doing matmult as a single matrix-matrix product\n"));
          break;
        case BV_MATMULT_MAT_SAVE:
          CHKERRQ(PetscViewerASCIIPrintf(viewer,"  mat_save is deprecated, use mat\n"));
          break;
      }
      if (bv->rrandom) {
        CHKERRQ(PetscViewerASCIIPrintf(viewer,"  generating random vectors independent of the number of processes\n"));
      }
      if (bv->ops->view) CHKERRQ((*bv->ops->view)(bv,viewer));
    } else {
      if (bv->ops->view) CHKERRQ((*bv->ops->view)(bv,viewer));
      else CHKERRQ(BVView_Default(bv,viewer));
    }
  } else {
    CHKERRQ((*bv->ops->view)(bv,viewer));
  }
  PetscFunctionReturn(0);
}

/*@C
   BVViewFromOptions - View from options

   Collective on BV

   Input Parameters:
+  bv   - the basis vectors context
.  obj  - optional object
-  name - command line option

   Level: intermediate

.seealso: BVView(), BVCreate()
@*/
PetscErrorCode BVViewFromOptions(BV bv,PetscObject obj,const char name[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(bv,BV_CLASSID,1);
  CHKERRQ(PetscObjectViewFromOptions((PetscObject)bv,obj,name));
  PetscFunctionReturn(0);
}

/*@C
   BVRegister - Adds a new storage format to the BV package.

   Not collective

   Input Parameters:
+  name     - name of a new user-defined BV
-  function - routine to create context

   Notes:
   BVRegister() may be called multiple times to add several user-defined
   basis vectors.

   Level: advanced

.seealso: BVRegisterAll()
@*/
PetscErrorCode BVRegister(const char *name,PetscErrorCode (*function)(BV))
{
  PetscFunctionBegin;
  CHKERRQ(BVInitializePackage());
  CHKERRQ(PetscFunctionListAdd(&BVList,name,function));
  PetscFunctionReturn(0);
}

PetscErrorCode BVAllocateWork_Private(BV bv,PetscInt s)
{
  PetscFunctionBegin;
  if (s>bv->lwork) {
    CHKERRQ(PetscFree(bv->work));
    CHKERRQ(PetscMalloc1(s,&bv->work));
    CHKERRQ(PetscLogObjectMemory((PetscObject)bv,(s-bv->lwork)*sizeof(PetscScalar)));
    bv->lwork = s;
  }
  PetscFunctionReturn(0);
}

#if defined(PETSC_USE_DEBUG)
/*
   SlepcDebugBVView - partially view a BV object, to be used from within a debugger.

     ini, end: columns to be viewed
     s: name of Matlab variable
     filename: optionally write output to a file
 */
PETSC_UNUSED PetscErrorCode SlepcDebugBVView(BV bv,PetscInt ini,PetscInt end,const char *s,const char *filename)
{
  PetscInt       N,m;
  PetscScalar    *array;

  PetscFunctionBegin;
  CHKERRQ(BVGetArray(bv,&array));
  CHKERRQ(BVGetSizes(bv,NULL,&N,&m));
  CHKERRQ(SlepcDebugViewMatrix(N,end-ini+1,array+ini*N,NULL,N,s,filename));
  CHKERRQ(BVRestoreArray(bv,&array));
  PetscFunctionReturn(0);
}
#endif
