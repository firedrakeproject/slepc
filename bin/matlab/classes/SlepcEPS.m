classdef SlepcEPS < PetscObject
%
%   SlepcEPS - a SLEPc eigenvalue solver object
%
%   Creation:
%     eps = SlepcEPS;
%     eps.SetType('krylovschur');
%     eps.SetOperators(A,B);
%     (optional) eps.SetProblemType(...);
%     eps.SetFromOptions;
%
  properties (Constant)
    HEP=1;
    GHEP=2;
    NHEP=3;
    GNHEP=4;
    PGNHEP=5;
    GHIEP=6;

    RITZ=1;
    HARMONIC=2;
    HARMONIC_RELATIVE=3;
    HARMONIC_RIGHT=4;
    HARMONIC_LARGEST=5;
    REFINED=6;
    REFINED_HARMONIC=7;

    LARGEST_MAGNITUDE=1;
    SMALLEST_MAGNITUDE=2;
    LARGEST_REAL=3;
    SMALLEST_REAL=4;
    LARGEST_IMAGINARY=5;
    SMALLEST_IMAGINARY=6;
    TARGET_MAGNITUDE=7;
    TARGET_REAL=8;
    TARGET_IMAGINARY=9;
    ALL=10;
    WHICH_USER=11;
  end
  methods
    function obj = SlepcEPS(pid,flag)
      if (nargin > 1) 
        %  SelpcEPS(pid,'pobj') uses an already existing SLEPc EPS object
        obj.pobj = pid;
        return
      end
      comm =  PETSC_COMM_SELF();
      [err,obj.pobj] = calllib('libslepc', 'EPSCreate', comm,0);PetscCHKERRQ(err);
    end
    function err = SetType(obj,name)
      err = calllib('libslepc', 'EPSSetType', obj.pobj,name);PetscCHKERRQ(err);
    end
    function err = SetFromOptions(obj)
      err = calllib('libslepc', 'EPSSetFromOptions', obj.pobj);PetscCHKERRQ(err);
    end
    function err = SetUp(obj)
      err = calllib('libslepc', 'EPSSetUp', obj.pobj);PetscCHKERRQ(err);
    end
    function err = Solve(obj)
      err = calllib('libslepc', 'EPSSolve', obj.pobj);PetscCHKERRQ(err);
    end
    function err = SetOperators(obj,A,B)
      if (nargin == 2) 
        err = calllib('libslepc', 'EPSSetOperators', obj.pobj,A.pobj,0);PetscCHKERRQ(err);
      else
        err = calllib('libslepc', 'EPSSetOperators', obj.pobj,A.pobj,B.pobj);PetscCHKERRQ(err);
      end
    end
    function err = SetProblemType(obj,t)
      err = calllib('libslepc', 'EPSSetProblemType', obj.pobj,t);PetscCHKERRQ(err);
    end
    function err = SetWhichEigenpairs(obj,t)
      err = calllib('libslepc', 'EPSSetWhichEigenpairs', obj.pobj,t);PetscCHKERRQ(err);
    end
    function err = SetExtraction(obj,t)
      err = calllib('libslepc', 'EPSSetExtraction', obj.pobj,t);PetscCHKERRQ(err);
    end
    function err = SetTolerances(obj,t,mx)
      if (nargin == 2) mx = 0; end
      err = calllib('libslepc', 'EPSSetTolerances', obj.pobj,t,mx);PetscCHKERRQ(err);
    end
    function err = SetDimensions(obj,nev,ncv,mpd)
      if (nargin < 3) ncv = 0; end
      if (nargin < 4) mpd = 0; end
      err = calllib('libslepc', 'EPSSetDimensions', obj.pobj,nev,ncv,mpd);PetscCHKERRQ(err);
    end
    function err = SetTarget(obj,t)
      err = calllib('libslepc', 'EPSSetTarget', obj.pobj,t);PetscCHKERRQ(err);
    end
    function [nconv,err] = GetConverged(obj)
      nconv = 0;
      [err,nconv] = calllib('libslepc', 'EPSGetConverged', obj.pobj,nconv);PetscCHKERRQ(err);
    end
    function [lambda,err] = GetEigenpair(obj,i)
      lambda = 0.0;
      img = 0.0;
      [err,lambda,img] = calllib('libslepc', 'EPSGetEigenpair', obj.pobj,i-1,lambda,img,0,0);PetscCHKERRQ(err);
      if img~=0.0, lambda = lambda+j*img; end
    end
    function [relerr,err] = ComputeRelativeError(obj,i)
      relerr = 0.0;
      [err,relerr] = calllib('libslepc', 'EPSComputeRelativeError', obj.pobj,i-1,relerr);PetscCHKERRQ(err);
    end
    function err = View(obj,viewer)
      if (nargin == 1)
        err = calllib('libslepc', 'EPSView', obj.pobj,0);PetscCHKERRQ(err);
      else
        err = calllib('libslepc', 'EPSView', obj.pobj,viewer.pobj);PetscCHKERRQ(err);
      end
    end
    function [st,err] = GetST(obj)
      [err,pid] = calllib('libslepc', 'EPSGetST', obj.pobj,0);PetscCHKERRQ(err);
      st = SlepcST(pid,'pobj');
    end
    function err = Destroy(obj)
      err = calllib('libslepc', 'EPSDestroy', obj.pobj);PetscCHKERRQ(err);
    end
  end
end

 
