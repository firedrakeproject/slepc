/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

#include "network.h"

const char *GraphTypes[] = {"sym","asym","bip",NULL};
const char *GraphWeights[] = {"unweighted","positive","posweighted","signed","multisigned","weighted","multiweighted","dynamic","multiposweighted",NULL};

PetscErrorCode GraphCreate(MPI_Comm comm,Graph *graph)
{
  PetscFunctionBeginUser;
  PetscCall(PetscNew(graph));
  (*graph)->comm = comm;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode GraphDestroy(Graph *graph)
{
  PetscFunctionBeginUser;
  if (!*graph) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(MatDestroy(&((*graph)->adjacency)));
  PetscCall(PetscFree(*graph));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode GraphPreload(Graph graph,char *filename)
{
  PetscInt    i,nval,src,dst;
  PetscBool   flg;
  PetscMPIInt rank;
  FILE        *file;
  char        gtype[64],gweight[64],line[PETSC_MAX_PATH_LEN];

  PetscFunctionBeginUser;
  PetscCallMPI(MPI_Comm_rank(graph->comm,&rank));
  if (rank==0) {
    PetscCall(PetscFOpen(PETSC_COMM_SELF,filename,"r",&file));
    /* process first line of the file */
    nval = fscanf(file,"%%%s%s\n",gtype,gweight);
    PetscCheck(nval==2,PETSC_COMM_SELF,PETSC_ERR_FILE_UNEXPECTED,"Badly formatted input file");
    for (i=0;i<(int)sizeof(GraphTypes);i++) {
      PetscCheck(GraphTypes[i],PETSC_COMM_SELF,PETSC_ERR_FILE_UNEXPECTED,"Unknown graph type %s",gtype);
      PetscCall(PetscStrcmp(gtype,GraphTypes[i],&flg));
      if (flg) { graph->type = (GraphType)i; break; }
    }
    for (i=0;i<(int)sizeof(GraphWeights);i++) {
      PetscCheck(GraphWeights[i],PETSC_COMM_SELF,PETSC_ERR_FILE_UNEXPECTED,"Unknown graph weight %s",gweight);
      PetscCall(PetscStrcmp(gweight,GraphWeights[i],&flg));
      if (flg) { graph->weight = (GraphWeight)i; break; }
    }
    /* skip second line of the file if it is a comment */
    if (!fgets(line,PETSC_MAX_PATH_LEN,file)) line[0] = 0;
    if (line[0]=='%') {
      if (!fgets(line,PETSC_MAX_PATH_LEN,file)) line[0] = 0;
    }
    graph->nedges = 1;
    nval = sscanf(line,"%" PetscInt_FMT "%" PetscInt_FMT,&src,&dst);
    PetscCheck(nval==2,PETSC_COMM_SELF,PETSC_ERR_FILE_UNEXPECTED,"Badly formatted input file");
    graph->nvertices = PetscMax(src,dst);
    /* read rest of file to count lines */
    while (fgets(line,PETSC_MAX_PATH_LEN,file)) {
      nval = sscanf(line,"%" PetscInt_FMT "%" PetscInt_FMT,&src,&dst);
      PetscCheck(nval==2,PETSC_COMM_SELF,PETSC_ERR_FILE_UNEXPECTED,"Badly formatted input file");
      graph->nedges++;
      graph->nvertices = PetscMax(graph->nvertices,PetscMax(src,dst));
    }
    PetscCall(PetscFClose(PETSC_COMM_SELF,file));
  }
  PetscCallMPI(MPI_Bcast(&graph->type,1,MPIU_INT,0,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Bcast(&graph->weight,1,MPIU_INT,0,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Bcast(&graph->nvertices,1,MPIU_INT,0,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Bcast(&graph->nedges,1,MPIU_INT,0,PETSC_COMM_WORLD));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode GraphPreallocate(Graph graph,char *filename)
{
  PetscInt i,nval,src,dst,Istart,Iend,*d_nnz,*o_nnz;
  FILE     *file;
  char     line[PETSC_MAX_PATH_LEN];

  PetscFunctionBeginUser;
  PetscCheck(graph->nvertices,PETSC_COMM_SELF,PETSC_ERR_ORDER,"Must call GraphPreload() first");
  PetscCall(MatDestroy(&graph->adjacency));
  PetscCall(MatCreate(graph->comm,&graph->adjacency));
  PetscCall(MatSetSizes(graph->adjacency,PETSC_DECIDE,PETSC_DECIDE,graph->nvertices,graph->nvertices));
  PetscCall(MatSetType(graph->adjacency,MATAIJ));
  PetscCall(MatGetOwnershipRange(graph->adjacency,&Istart,&Iend));
  PetscCall(PetscCalloc2(Iend-Istart,&d_nnz,Iend-Istart,&o_nnz));

  /* all process read the file */
  PetscCall(PetscFOpen(PETSC_COMM_SELF,filename,"r",&file));
  nval = fscanf(file,"%%%*s%*s\n");   /* first line of the file */
  if (!fgets(line,PETSC_MAX_PATH_LEN,file)) line[0] = 0;
  if (line[0]=='%') { /* skip second line of the file if it is a comment */
    if (!fgets(line,PETSC_MAX_PATH_LEN,file)) line[0] = 0;
  }
  nval = sscanf(line,"%" PetscInt_FMT "%" PetscInt_FMT,&src,&dst);
  PetscCheck(nval==2,PETSC_COMM_SELF,PETSC_ERR_FILE_UNEXPECTED,"Badly formatted input file");
  src--; dst--;     /* adjust from 1-based to 0-based */
  if (src>=Istart && src<Iend) {
    if (dst>=Istart && dst<Iend) d_nnz[src-Istart]++;
    else o_nnz[src-Istart]++;
  }
  /* read rest of file */
  while (fgets(line,PETSC_MAX_PATH_LEN,file)) {
    nval = sscanf(line,"%" PetscInt_FMT "%" PetscInt_FMT,&src,&dst);
    PetscCheck(nval==2,PETSC_COMM_SELF,PETSC_ERR_FILE_UNEXPECTED,"Badly formatted input file");
    src--; dst--;     /* adjust from 1-based to 0-based */
    if (src>=Istart && src<Iend) {
      if (dst>=Istart && dst<Iend) d_nnz[src-Istart]++;
      else o_nnz[src-Istart]++;
    }
  }
  PetscCall(PetscFClose(PETSC_COMM_SELF,file));

  for (i=Istart;i<Iend;i++) d_nnz[i-Istart]++;  /* diagonal entries */
  PetscCall(MatSeqAIJSetPreallocation(graph->adjacency,0,d_nnz));
  PetscCall(MatMPIAIJSetPreallocation(graph->adjacency,0,d_nnz,0,o_nnz));
  PetscCall(PetscFree2(d_nnz,o_nnz));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode GraphLoadUnweighted(Graph graph,char *filename)
{
  PetscInt i,nval,src,dst,Istart,Iend;
  FILE     *file;
  char     line[PETSC_MAX_PATH_LEN];

  PetscFunctionBeginUser;
  PetscCheck(graph->adjacency,PETSC_COMM_SELF,PETSC_ERR_ORDER,"Must call GraphPreallocate() first");
  PetscCall(MatGetOwnershipRange(graph->adjacency,&Istart,&Iend));
  /* all process read the file */
  PetscCall(PetscFOpen(PETSC_COMM_SELF,filename,"r",&file));
  nval = fscanf(file,"%%%*s%*s\n");   /* first line of the file */
  if (!fgets(line,PETSC_MAX_PATH_LEN,file)) line[0] = 0;
  if (line[0]=='%') { /* skip second line of the file if it is a comment */
    if (!fgets(line,PETSC_MAX_PATH_LEN,file)) line[0] = 0;
  }
  nval = sscanf(line,"%" PetscInt_FMT "%" PetscInt_FMT,&src,&dst);
  PetscCheck(nval==2,PETSC_COMM_SELF,PETSC_ERR_FILE_UNEXPECTED,"Badly formatted input file");
  src--; dst--;     /* adjust from 1-based to 0-based */
  if (src>=Istart && src<Iend) PetscCall(MatSetValue(graph->adjacency,src,dst,1.0,INSERT_VALUES));
  /* read rest of file */
  while (fgets(line,PETSC_MAX_PATH_LEN,file)) {
    nval = sscanf(line,"%" PetscInt_FMT "%" PetscInt_FMT,&src,&dst);
    PetscCheck(nval==2,PETSC_COMM_SELF,PETSC_ERR_FILE_UNEXPECTED,"Badly formatted input file");
    src--; dst--;     /* adjust from 1-based to 0-based */
    if (src>=Istart && src<Iend) PetscCall(MatSetValue(graph->adjacency,src,dst,1.0,INSERT_VALUES));
  }
  PetscCall(PetscFClose(PETSC_COMM_SELF,file));
  for (i=Istart;i<Iend;i++) PetscCall(MatSetValue(graph->adjacency,i,i,0.0,INSERT_VALUES));  /* diagonal entries */
  PetscCall(MatAssemblyBegin(graph->adjacency,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(graph->adjacency,MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}
