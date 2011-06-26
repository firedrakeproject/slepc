classdef SlepcSVD < PetscObject
%
%   SlepcSVD - a SLEPc singular value solver object
%
%   Creation:
%     svd = SlepcSVD;
%     svd.SetType('cross');
%     svd.SetOperator(A);
%     svd.SetFromOptions;
%
  properties (Constant)
    SVD_TRANSPOSE_EXPLICIT=0;
    SVD_TRANSPOSE_IMPLICIT=1;

    LARGEST=0;
    SMALLEST=1;
  end
  methods
    function obj = SlepcSVD(pid,flag)
      if (nargin > 1) 
        %  SelpcSVD(pid,'pobj') uses an already existing SLEPc SVD object
        obj.pobj = pid;
        return
      end
      comm =  PETSC_COMM_SELF();
      [err,obj.pobj] = calllib('libslepc', 'SVDCreate', comm,0);PetscCHKERRQ(err);
    end
    function err = SetType(obj,name)
      err = calllib('libslepc', 'SVDSetType', obj.pobj,name);PetscCHKERRQ(err);
    end
    function err = SetFromOptions(obj)
      err = calllib('libslepc', 'SVDSetFromOptions', obj.pobj);PetscCHKERRQ(err);
    end
    function err = SetUp(obj)
      err = calllib('libslepc', 'SVDSetUp', obj.pobj);PetscCHKERRQ(err);
    end
    function err = Solve(obj)
      err = calllib('libslepc', 'SVDSolve', obj.pobj);PetscCHKERRQ(err);
    end
    function err = SetOperator(obj,A)
      err = calllib('libslepc', 'SVDSetOperator', obj.pobj,A.pobj);PetscCHKERRQ(err);
    end
    function err = SetTransposeMode(obj,t)
      err = calllib('libslepc', 'SVDSetTransposeMode', obj.pobj,t);PetscCHKERRQ(err);
    end
    function err = SetWhichSingularTriplets(obj,t)
      err = calllib('libslepc', 'SVDSetWhichSingularTriplets', obj.pobj,t);PetscCHKERRQ(err);
    end
    function err = SetTolerances(obj,t,mx)
      if (nargin == 2) mx = 0; end
      err = calllib('libslepc', 'SVDSetTolerances', obj.pobj,t,mx);PetscCHKERRQ(err);
    end
    function err = SetDimensions(obj,nev,ncv,mpd)
      if (nargin < 3) ncv = 0; end
      if (nargin < 4) mpd = 0; end
      err = calllib('libslepc', 'SVDSetDimensions', obj.pobj,nev,ncv,mpd);PetscCHKERRQ(err);
    end
    function [nconv,err] = GetConverged(obj)
      nconv = 0;
      [err,nconv] = calllib('libslepc', 'SVDGetConverged', obj.pobj,nconv);PetscCHKERRQ(err);
    end
    function [sigma,err] = GetSingularTriplet(obj,i)
      sigma = 0.0;
      [err,sigma] = calllib('libslepc', 'SVDGetSingularTriplet', obj.pobj,i-1,sigma,0,0);PetscCHKERRQ(err);
    end
    function [relerr,err] = ComputeRelativeError(obj,i)
      relerr = 0.0;
      [err,relerr] = calllib('libslepc', 'SVDComputeRelativeError', obj.pobj,i-1,relerr);PetscCHKERRQ(err);
    end
    function err = View(obj,viewer)
      if (nargin == 1)
        err = calllib('libslepc', 'SVDView', obj.pobj,0);PetscCHKERRQ(err);
      else
        err = calllib('libslepc', 'SVDView', obj.pobj,viewer.pobj);PetscCHKERRQ(err);
      end
    end
    function err = Destroy(obj)
      err = calllib('libslepc', 'SVDDestroy', obj.pobj);PetscCHKERRQ(err);
    end
  end
end

 
