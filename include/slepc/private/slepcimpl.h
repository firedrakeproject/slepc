/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

#if !defined(SLEPCIMPL_H)
#define SLEPCIMPL_H

#include <slepcsys.h>
#include <petsc/private/petscimpl.h>

/* SUBMANSEC = sys */

SLEPC_INTERN PetscBool SlepcBeganPetsc;

/*@C
    SlepcHeaderCreate - Creates a SLEPc object

    Input Parameters:
+   classid - the classid associated with this object
.   class_name - string name of class; should be static
.   descr - string containing short description; should be static
.   mansec - string indicating section in manual pages; should be static
.   comm - the MPI Communicator
.   destroy - the destroy routine for this object
-   view - the view routine for this object

    Output Parameter:
.   h - the newly created object

    Note:
    This is equivalent to PetscHeaderCreate but makes sure that SlepcInitialize
    has been called.

    Level: developer
@*/
#define SlepcHeaderCreate(h,classid,class_name,descr,mansec,comm,destroy,view) \
    ((!SlepcInitializeCalled && \
    PetscError(comm,__LINE__,PETSC_FUNCTION_NAME,__FILE__,1,PETSC_ERROR_INITIAL, \
    "Must call SlepcInitialize instead of PetscInitialize to use SLEPc classes")) ||  \
    PetscHeaderCreate(h,classid,class_name,descr,mansec,comm,destroy,view))

/* context for monitors of type XXXMonitorConverged */
struct _n_SlepcConvMon {
  void     *ctx;
  PetscInt oldnconv;  /* previous value of nconv */
};

/*
  SlepcPrintEigenvalueASCII - Print an eigenvalue on an ASCII viewer.
*/
static inline PetscErrorCode SlepcPrintEigenvalueASCII(PetscViewer viewer,PetscScalar eigr,PetscScalar eigi)
{
  PetscReal      re,im;

  PetscFunctionBegin;
#if defined(PETSC_USE_COMPLEX)
  re = PetscRealPart(eigr);
  im = PetscImaginaryPart(eigr);
#else
  re = eigr;
  im = eigi;
#endif
  /* print zero instead of tiny value */
  if (PetscAbs(im) && PetscAbs(re)/PetscAbs(im)<PETSC_SMALL) re = 0.0;
  if (PetscAbs(re) && PetscAbs(im)/PetscAbs(re)<PETSC_SMALL) im = 0.0;
  /* print as real if imaginary part is zero */
  if (im!=0.0) PetscCall(PetscViewerASCIIPrintf(viewer,"%.5f%+.5fi",(double)re,(double)im));
  else PetscCall(PetscViewerASCIIPrintf(viewer,"%.5f",(double)re));
  PetscFunctionReturn(0);
}

/*
  SlepcViewEigenvector - Outputs an eigenvector xr,xi to a viewer.
  In complex scalars only xr is written.
  The name of xr,xi is set before writing, based on the label, the index, and the name of obj.
*/
static inline PetscErrorCode SlepcViewEigenvector(PetscViewer viewer,Vec xr,Vec xi,const char *label,PetscInt index,PetscObject obj)
{
  size_t         count;
  char           vname[30];
  const char     *pname;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetName(obj,&pname));
  PetscCall(PetscSNPrintfCount(vname,sizeof(vname),"%s%s",&count,label,PetscDefined(USE_COMPLEX)?"":"r"));
  count--;
  PetscCall(PetscSNPrintf(vname+count,sizeof(vname)-count,"%" PetscInt_FMT "_%s",index,pname));
  PetscCall(PetscObjectSetName((PetscObject)xr,vname));
  PetscCall(VecView(xr,viewer));
#if !defined(PETSC_USE_COMPLEX)
  vname[count-1] = 'i';
  PetscCall(PetscObjectSetName((PetscObject)xi,vname));
  PetscCall(VecView(xi,viewer));
#endif
  PetscFunctionReturn(0);
}

/* Macros for strings with different value in real and complex */
#if defined(PETSC_USE_COMPLEX)
#define SLEPC_STRING_HERMITIAN "hermitian"
#else
#define SLEPC_STRING_HERMITIAN "symmetric"
#endif

/* Private functions that are shared by several classes */
SLEPC_EXTERN PetscErrorCode SlepcBasisReference_Private(PetscInt,Vec*,PetscInt*,Vec**);
SLEPC_EXTERN PetscErrorCode SlepcBasisDestroy_Private(PetscInt*,Vec**);
SLEPC_EXTERN PetscErrorCode SlepcMonitorMakeKey_Internal(const char[],PetscViewerType,PetscViewerFormat,char[]);
SLEPC_EXTERN PetscErrorCode PetscViewerAndFormatCreate_Internal(PetscViewer,PetscViewerFormat,void*,PetscViewerAndFormat**);

SLEPC_INTERN PetscErrorCode SlepcCitationsInitialize(void);
SLEPC_INTERN PetscErrorCode SlepcInitialize_DynamicLibraries(void);
SLEPC_INTERN PetscErrorCode SlepcInitialize_Packages(void);

/* Definitions needed to work with CUDA kernels */
#if defined(PETSC_HAVE_CUDA)
#include <petscdevice_cuda.h>

#define X_AXIS 0
#define Y_AXIS 1

#define SLEPC_TILE_SIZE_X  32
#define SLEPC_BLOCK_SIZE_X 128
#define SLEPC_TILE_SIZE_Y  32
#define SLEPC_BLOCK_SIZE_Y 128

static inline PetscErrorCode SlepcKernelSetGrid1D(PetscInt rows,dim3 *dimGrid,dim3 *dimBlock,PetscInt *dimGrid_xcount)
{
  int                   card;
  struct cudaDeviceProp devprop;

  PetscFunctionBegin;
  PetscCallCUDA(cudaGetDevice(&card));
  PetscCallCUDA(cudaGetDeviceProperties(&devprop,card));
  *dimGrid_xcount = 1;

  /* X axis */
  dimGrid->x  = 1;
  dimBlock->x = SLEPC_BLOCK_SIZE_X;
  if (rows>SLEPC_BLOCK_SIZE_X) dimGrid->x = (rows+SLEPC_BLOCK_SIZE_X-1)/SLEPC_BLOCK_SIZE_X;
  else dimBlock->x = rows;
  if (dimGrid->x>(unsigned)devprop.maxGridSize[X_AXIS]) {
    *dimGrid_xcount = (dimGrid->x+(devprop.maxGridSize[X_AXIS]-1))/devprop.maxGridSize[X_AXIS];
    dimGrid->x = devprop.maxGridSize[X_AXIS];
  }
  PetscFunctionReturn(0);
}

static inline PetscErrorCode SlepcKernelSetGrid2DTiles(PetscInt rows,PetscInt cols,dim3 *dimGrid,dim3 *dimBlock,PetscInt *dimGrid_xcount,PetscInt *dimGrid_ycount)
{
  int                   card;
  struct cudaDeviceProp devprop;

  PetscFunctionBegin;
  PetscCallCUDA(cudaGetDevice(&card));
  PetscCallCUDA(cudaGetDeviceProperties(&devprop,card));
  *dimGrid_xcount = *dimGrid_ycount = 1;

  /* X axis */
  dimGrid->x  = 1;
  dimBlock->x = SLEPC_BLOCK_SIZE_X;
  if (rows>SLEPC_BLOCK_SIZE_X*SLEPC_TILE_SIZE_X) dimGrid->x = (rows+SLEPC_BLOCK_SIZE_X*SLEPC_TILE_SIZE_X-1)/(SLEPC_BLOCK_SIZE_X*SLEPC_TILE_SIZE_X);
  else dimBlock->x = (rows+SLEPC_TILE_SIZE_X-1)/SLEPC_TILE_SIZE_X;
  if (dimGrid->x>(unsigned)devprop.maxGridSize[X_AXIS]) {
    *dimGrid_xcount = (dimGrid->x+(devprop.maxGridSize[X_AXIS]-1))/devprop.maxGridSize[X_AXIS];
    dimGrid->x = devprop.maxGridSize[X_AXIS];
  }

  /* Y axis */
  dimGrid->y  = 1;
  dimBlock->y = SLEPC_BLOCK_SIZE_Y;
  if (cols>SLEPC_BLOCK_SIZE_Y*SLEPC_TILE_SIZE_Y) dimGrid->y = (cols+SLEPC_BLOCK_SIZE_Y*SLEPC_TILE_SIZE_Y-1)/(SLEPC_BLOCK_SIZE_Y*SLEPC_TILE_SIZE_Y);
  else dimBlock->y = (cols+SLEPC_TILE_SIZE_Y-1)/SLEPC_TILE_SIZE_Y;
  if (dimGrid->y>(unsigned)devprop.maxGridSize[Y_AXIS]) {
    *dimGrid_ycount = (dimGrid->y+(devprop.maxGridSize[Y_AXIS]-1))/devprop.maxGridSize[Y_AXIS];
    dimGrid->y = devprop.maxGridSize[Y_AXIS];
  }
  PetscFunctionReturn(0);
}
#undef X_AXIS
#undef Y_AXIS
#endif

#endif
