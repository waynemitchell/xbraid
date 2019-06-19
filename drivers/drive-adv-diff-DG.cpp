// Copyright (c) 2013, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory. Written by
// Jacob Schroder, Rob Falgout, Tzanio Kolev, Ulrike Yang, Veselin
// Dobrev, et al. LLNL-CODE-660355. All rights reserved.
// Modified by Wayne Mitchell to include AIR AMG as the solver for implicit methods.
//
// This file is part of XBraid. For support, post issues to the XBraid Github page.
//
// This program is free software; you can redistribute it and/or modify it under
// the terms of the GNU General Public License (as published by the Free Software
// Foundation) version 2.1 dated February 1999.
//
// This program is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or FITNESS FOR A
// PARTICULAR PURPOSE. See the terms and conditions of the GNU General Public
// License for more details.
//
// You should have received a copy of the GNU Lesser General Public License along
// with this program; if not, write to the Free Software Foundation, Inc., 59
// Temple Place, Suite 330, Boston, MA 02111-1307 USA

// Driver:        drive-adv-diff-DG.cpp
//
// Interface:     C++, through MFEM 
// 
// Requires:      MFEM (AIR branch), Hypre, Metis and GlVis
//                Modify Makefile to point to metis, mfem and hypre libraries
//
// Compile with:  make drive-adv-diff-DG
//
// Help with:     drive-adv-diff-DG -help
//
// Sample runs:   mpirun -np 4 drive-adv-diff-DG -rp 0 -rs 1 -nt 100 --ode-solver 11
//
// Description:   Solves (a) scalar ODE problems, and (b) the 2D/3D heat equation

#include "braid_mfem.hpp"
#include "mfem_arnoldi.hpp"
#include "hypre_extra.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace hypre;

// Choice for the problem setup. The fluid velocity, initial condition and
// inflow boundary condition are chosen based on this parameter.
int problem;
Vector bb_min, bb_max;

// Source function
double source_function(Vector &x, double t);

// Velocity coefficient
void velocity_function(const Vector &x, Vector &v);

// Initial condition
double u0_function(Vector &x);

// Inflow boundary condition
double inflow_function(Vector &x);

struct DGAdvectionOptions : public BraidOptions
{
   bool   plot_mode;
   int    problem;
   int    n_x;
   double diffusion;
   int    order;
   int    ode_solver_type;
   int    basis_type;
   bool   lump_mass;
   double dt; // derived from t_start, t_final, and num_time_steps
   bool   krylov_coarse;
   int    krylov_size;
   bool   init_rand;
   int    prec_type; // passed down to FE_Evolution
   int    diss_oper_type;
   bool   dirichlet_bcs;

   const char *vishost;
   int         visport;

   int vis_time_steps;
   int vis_braid_steps;
   bool vis_screenshots;

   bool write_matrices;

   DGAdvectionOptions(int argc, char *argv[]);
};

/** A time-dependent operator for the right-hand side of the ODE. The DG weak
    form of du/dt = v.grad(u) is M du/dt = K u + b, where M and K are the mass
    and advection matrices, and b describes the flow on the boundary. This can
    be written as a general ODE, du/dt = M^{-1} (K u + b), and this class is
    used to evaluate the right-hand side. */
class FE_Evolution : public TimeDependentOperator
{
private:
   DGAdvectionOptions &options;
   HypreParMatrix &M, &K;
   HypreParVector **b;
   HypreSmoother M_prec;
   CGSolver M_solver;

   mutable Array<double> dts;
   mutable Array<HypreParMatrix*> B; // B = M - dt*K, for ImplicitSolve
   mutable Array<HypreParMatrix*> B_s; // Block inverse scaled version of B
   mutable Array<Operator*> B_prec;
   mutable Array<Solver*> B_solver;
   int prec_type;
   int blocksize;

   mutable Vector z; // auxiliary vector

   int GetDtIndex(double dt) const;

public:
   FE_Evolution(HypreParMatrix &_M, HypreParMatrix &_K, HypreParVector **_b, DGAdvectionOptions &_options);

   /** 0 - HypreParaSails, 1 - HypreBoomerAMG, 2 - UMFPackSolver */
   void SetPreconditionerType(int type) { prec_type = type; }

   virtual void Mult(const Vector &x, Vector &y) const;

   /** Solve the equation: k = f(x + dt*k, t), for the unknown k.
       For this class the equation becomes:
          k = f(x + dt*k, t) = M^{-1} (K (x + dt*k) + b),
       or
          k = (M - dt*K)^{-1} (K x + b). */
   virtual void ImplicitSolve(const double dt, const Vector &x, Vector &k);

   /// Compute the M-inner product of x and y: y^t.M.x
   double InnerProduct(const Vector &x, const Vector &y) const;

   virtual ~FE_Evolution();
};

class DGAdvectionApp : public MFEMBraidApp
{
protected:
   DGAdvectionOptions &options;

   // !!! Debug: switching to continuous Galerkin
   DG_FECollection dg_fe_coll;
   H1_FECollection h1_fe_coll;

   FunctionCoefficient source;
   VectorFunctionCoefficient velocity;
   FunctionCoefficient inflow;
   FunctionCoefficient u0;
   // ConstantCoefficient diff;

   // Data for all discretization levels
   Array<HypreParMatrix *> M; // mass matrices
   Array<HypreParMatrix *> K; // advection matrices
   Array<HypreParVector *> B; // r.h.s. vectors

   Array<ParBilinearForm *> m;
   Array<ParBilinearForm *> k;
   Array<ParLinearForm *> b;

   int Step_calls_counter, Norm_calls_counter;

   // Allocate data structures for the given number of spatial levels. Used by
   // InitMultilevelApp.
   virtual void AllocLevels(int num_levels);

   // Construct the ParFiniteElmentSpace for the given mesh. Used by
   // InitMultilevelApp.
   virtual ParFiniteElementSpace *ConstructFESpace(ParMesh *pmesh);

   // Assuming mesh[l] and fe_space[l] are set, initialize, ode[l], solver[l],
   // and max_dt[l]. Used by InitMultilevelApp.
   virtual void InitLevel(int l);

public:
   DGAdvectionApp(DGAdvectionOptions &opts, MPI_Comm comm_t_, ParMesh *pmesh);
   
   SpaceTimeMeshInfo MeshInfo;

   // braid_Vector == BraidVector*
   virtual int Step(braid_Vector    u_,
                    braid_Vector    u_stop_,
                    braid_Vector    fstop_,
                    BraidStepStatus &pstatus);

   virtual int SpatialNorm(braid_Vector  u_,
                           double       *norm_ptr);

   virtual ~DGAdvectionApp();

   void PrintStats(MPI_Comm comm);
   void PlotMode();
};


int main(int argc, char *argv[])
{
   MPI_Init(&argc, &argv);

   MPI_Comm comm = MPI_COMM_WORLD;
   int myid;
   MPI_Comm_rank(comm, &myid);

   int precision = 8;
   cout.precision(precision);

   // Parse command line options
   DGAdvectionOptions opts(argc, argv);
   // Check for errors
   if (!opts.Good())
   {
      if (myid == 0)
      {
         opts.PrintUsage(cout);
      }
      MPI_Finalize();
      return 1;
   }
   // Print the used options
   if (myid == 0)
   {
      opts.PrintOptions(cout);
   }
   problem = opts.problem;

   // Load the mesh and refine it in each processor (serial refinement)
   Mesh *mesh = opts.LoadMeshAndSerialRefine();
   if (!mesh)
   {
      if (!strcmp(opts.mesh_file,"square")) mesh = new Mesh(opts.n_x, opts.n_x, Element::QUADRILATERAL);
      else if (!strcmp(opts.mesh_file,"cube")) mesh = new Mesh(opts.n_x, opts.n_x, opts.n_x, Element::QUADRILATERAL);
      // if (myid == 0)
      // {
      //    cerr << "\nError loading mesh file: " << opts.mesh_file
      //         << '\n' << endl;
      // }
      // MPI_Finalize();
      // return 2;
   }

   // If the mesh is NURBS, convert it to curved mesh
   if (mesh->NURBSext)
   {
      mesh->SetCurvature(std::max(opts.order, 1));
   }
   // !!! Changed: getting bounding box info
   mesh->GetBoundingBox(bb_min, bb_max, max(opts.order, 1));

   // Split comm (MPI_COMM_WORLD) into spatial and temporal communicators
   MPI_Comm comm_x, comm_t;
   BraidUtil util;
   util.SplitCommworld(&comm, opts.num_procs_x, &comm_x, &comm_t);

   // Partition the mesh accross the spatial communicator
   ParMesh *pmesh = new ParMesh(comm_x, *mesh);
   delete mesh;

   // Create and initialize a DGAdvectionApp (including parallel refinement)
   DGAdvectionApp app(opts, comm_t, pmesh);

   // Run a Braid simulation
   BraidCore core(comm, &app);
   opts.SetBraidCoreOptions(core);

   if (opts.plot_mode)
   {
      app.PlotMode();
   }
   else
   {
      core.Drive();
      app.PrintStats(comm);
      
      // Print the residual convergence factors to file
      braid_Int num_rnorm = 100; 
      braid_Real *rnorms = (braid_Real*) calloc(num_rnorm, sizeof(braid_Real));
      core.GetRNorms(&num_rnorm, rnorms);
      ofstream outfile;
      outfile.open("res_prob" + to_string(opts.problem) + "_size" + to_string(opts.n_x) + "_k" + to_string(opts.cfactor) + "_solver" + to_string(opts.ode_solver_type) + ".txt");
      outfile << rnorms[0] << " " << 1.0 << endl;
      for (auto i = 0; i < num_rnorm-1; i++)
      {
         braid_Real cf = rnorms[i+1]/rnorms[i];
         outfile << rnorms[i+1] << " " << cf << endl;
      }
      outfile.close();
   }

   MPI_Finalize();
   return 0;
}


// Implementation of class FE_Evolution
FE_Evolution::FE_Evolution(HypreParMatrix &_M, HypreParMatrix &_K, HypreParVector **_b, DGAdvectionOptions &_options)
   : TimeDependentOperator(_M.Height()),
     options(_options), M(_M), K(_K), b(_b), M_solver(M.GetComm()), z(_M.Height())
{
   M_prec.SetType(HypreSmoother::Jacobi);
   M_solver.SetPreconditioner(M_prec);
   M_solver.SetOperator(M);

   M_solver.iterative_mode = false;
   M_solver.SetRelTol(1e-9);
   M_solver.SetAbsTol(0.0);
   M_solver.SetMaxIter(100);
   M_solver.SetPrintLevel(0);

   // DG block size given by (FEorder+1)^2 on square meshes.
   blocksize = (options.order+1)*(options.order+1);
}

FE_Evolution::~FE_Evolution()
{
   for (int i = dts.Size()-1; i >= 0; i--)
   {
      delete B_solver[i];
      delete B_prec[i];
      delete B[i];
      // !!! Why does the delete below cause memory error?
      // if (B_s[i]) delete B_s[i];
   }
}

int FE_Evolution::GetDtIndex(double dt) const
{
   for (int i = 0; i < dts.Size(); i++)
   {
      if (std::abs(dts[i]-dt) < 1e-10*dt)
      {
         return i;
      }
   }
   // cout << "\nConstructing (M - dt K) and solver for dt = " << dt << endl;
   dts.Append(dt);
   B.Append(new HypreParMatrix(hypre_ParCSRMatrixAdd(M, K)));
   HypreParMatrix &B_new = *B.Last();
   hypre_ParCSRMatrixSetConstantValues(B_new, 0.0);
   hypre_ParCSRMatrixSum(B_new, 1.0, M);
   hypre_ParCSRMatrixSum(B_new, -dt, K);

   B_s.Append(new HypreParMatrix(B_new));
   HypreParMatrix &B_s_new = *B_s.Last();
   if (options.problem != 0) BlockInvScal(&B_new, &B_s_new, NULL, NULL, blocksize, 0);

   int print_level = 0;
   HypreBoomerAMG *AMG_solver = new HypreBoomerAMG(B_s_new);
   AMG_solver->SetMaxLevels(50);   

   if (options.problem != 0) {
      AMG_solver->SetLAIROptions(1.5, "", "FFC", 0.1, 0.01, 0.0,
      100, 3, 0.0, 10, -1, 1);
      // 100, 3, 0.0, 6, -1, 1);
      AMG_solver->SetPrintLevel(print_level);
   }
   else {
      AMG_solver->SetPrintLevel(print_level);
      AMG_solver->SetInterpolation(0);
      AMG_solver->SetCoarsening(6);
      AMG_solver->SetAggressiveCoarsening(1);
   }

   int use_gmres = 1;
   if (use_gmres) {
      HypreGMRES *GMRES_solver = new HypreGMRES(B_s_new);
      GMRES_solver->SetTol(1e-12);
      GMRES_solver->SetMaxIter(100);
      GMRES_solver->SetPrintLevel(print_level);
      GMRES_solver->SetPreconditioner(*AMG_solver);
      GMRES_solver->iterative_mode = false;
      B_prec.Append(AMG_solver);
      B_solver.Append(GMRES_solver);
   }
   else {
      AMG_solver->SetTol(1e-12);
      AMG_solver->SetMaxIter(100);
      B_solver.Append(AMG_solver);
   }

   return dts.Size()-1;
}

void FE_Evolution::Mult(const Vector &x, Vector &y) const
{
   // y = M^{-1} (K x + b)
   K.Mult(x, z);
   z += **b;
   M_solver.Mult(z, y);
}

void FE_Evolution::ImplicitSolve(const double dt, const Vector &x, Vector &k)
{
   // k = (M - dt*K)^{-1} (K x + b)
   int i = GetDtIndex(dt);
   K.Mult(x, z);
   z += **b;

   if (options.problem == 0)
   {
      B_solver[i]->Mult(z, k);
   }
   else
   {
      HypreParVector b_s;
      BlockInvScal(B[i], NULL, &z, &b_s, blocksize, 2);
      B_solver[i]->Mult(b_s, k);
   }

   if (HYPRE_GetError())
   {
      MFEM_WARNING("HYPRE error = " << HYPRE_GetError());
      HYPRE_ClearAllErrors();
   }
   // cout << "proc[" << K.GetComm() << "]: " << _MFEM_FUNC_NAME << endl;
}

double FE_Evolution::InnerProduct(const Vector &x, const Vector &y) const
{
   M.Mult(x, z);
   double local_dot = (y * z), global_dot;
   MPI_Allreduce(&local_dot, &global_dot, 1, MPI_DOUBLE, MPI_SUM, M.GetComm());
   return global_dot;
}


DGAdvectionOptions::DGAdvectionOptions(int argc, char *argv[])
   : BraidOptions(argc, argv)
{
   // set defaults for inherited options
   t_start        = 0.0;
   t_final        = 10.0;
   num_time_steps = 1000;
   num_procs_x    = 1;
   n_x            = 2;

   // set defaults for the (optional) mesh/refinement inherited options
   mesh_file       = "/home/nfs/wmitchell/Documents/mfem/data/periodic-square.mesh";
   // mesh_file       = "square";
   ser_ref_levels  = 0;
   par_ref_levels  = 0;
   AddMeshOptions();

   // set defaults for the DGAdvection-specific options
   plot_mode       = false;
   problem         = 0;
   diffusion       = 0.0;
   diss_oper_type  = 0; // 0 - diffusion (S), 1 - diffusion squared (S^2)
   // order           = 3;
   
   // !!! Debug: changing to continuous galerkin order 1
   order = 1;

   ode_solver_type = 4;
   basis_type      = 0;
   lump_mass       = false;
   krylov_coarse   = false;
   krylov_size     = 4;
   init_rand       = false;
   prec_type       = 0; // see FE_Evolution::SetPreconditionerType()
   vishost         = "localhost";
   visport         = 19916;
   vis_time_steps  = 0;
   vis_braid_steps = 0;
   vis_screenshots = false;
   write_matrices  = false;
   dirichlet_bcs   = true;

   AddOption(&plot_mode, "-pltmode", "--plot-mode",
             "-npltmode", "--no-plot-mode", "Enable/disable "
             " plot mode (if enabled, does not run any solve, just reads from file and plots).");
   AddOption(&problem, "-p", "--problem",
             "Problem setup to use. See options in velocity_function().");
   AddOption(&diffusion, "-diff", "--diffusion", "Diffusion coefficient.");
   AddOption(&diss_oper_type, "-diss", "--dissipation-type",
             "Dissipation operator type: 0 - diffusion, 1 - diffusion squared");
   AddOption(&order, "-o", "--order",
             "Order (degree) of the finite elements.");
   AddOption(&ode_solver_type, "-s", "--ode-solver",
             "ODE solver: 1 - Forward Euler, 2 - RK2 SSP, 3 - RK3 SSP,"
             " 4 - RK4, 6 - RK6,\n"
             "\t11 - Backward Euler, 12 - SDIRK-2, 13 - SDIRK-3");
   AddOption(&basis_type, "-b", "--basis-type",
             "DG basis type: 0 - Nodal Gauss-Legendre, 1 - Nodal Gauss-Lobatto,"
             " 2 - Positive");
   AddOption(&lump_mass, "-lump", "--lump-mass-matrix", "-dont-lump",
             "--dont-lump-mass-matrix", "Enable/disable lumping of the mass"
             " matrix.");
   AddOption(&krylov_coarse, "-kc", "--krylov-coarse", "-dont-kc",
             "--dont-krylov-coarse", "Enable/disable use of Arnoldi-based coarse-grid"
             " time-stepping.");
   AddOption(&krylov_size, "-ks", "--krylov-size", "Set size of the Krylov space for the"
                           " Arnoldi-based coarse-grid time-stepping.");
   AddOption(&init_rand, "-rand", "--random-init", "-zero", "--zero-init",
             "How to initialize vectors: with random numbers or zeros");
   AddOption(&prec_type, "-prec", "--preconditioner",
             "Preconditioner type (for implicit time stepping):\n\t"
             "0 - HypreParaSails, 1 - HypreBoomerAMG, 2 - UMFPACK (px=1)");
   AddOption(&vishost, "-vh", "--visualization-host",
             "Set the GLVis host.");
   AddOption(&visport, "-vp", "--visualization-port",
             "Set the GLVis port.");
   AddOption(&n_x, "-nx", "--nx",
             "Set number of elements in the x direction.");
   AddOption(&vis_time_steps, "-vts", "--visualize-time-steps",
             "Visualize every n-th time step (0:final only).");
   AddOption(&vis_braid_steps, "-vbs", "--visualize-braid-steps",
             "Visualize every n-th Braid step (0:final only).");
   AddOption(&vis_screenshots, "-vss", "--vis-screenshots",
             "-no-vss", "--vis-no-screenshots", "Enable/disable the saving of"
             " screenshots of the visualized data.");
   AddOption(&write_matrices, "-wm", "--write-matrices",
             "-dont-wm", "--dont-write-matrices", "Enable/disable the saving of"
             " the fine level mass (M) and advection (K) matrices.");
   AddOption(&dirichlet_bcs, "-dbc", "--dirichlet-bcs",
             "-ndbc", "--no-dirichlet-bcs", "Enable/disable "
             "Dirichlet boundary conditions.");

   Parse();

   if (problem == 0) diffusion = 1.0;
   if (problem % 10 == 1) diffusion = 1.0;
   if (problem % 10 == 2) diffusion = 0.1;
   if (problem % 10 == 3) diffusion = 0.01;
   if (problem % 10 == 4) diffusion = 0.001;

   dt = (t_final - t_start) / num_time_steps;
}


DGAdvectionApp::DGAdvectionApp(
      DGAdvectionOptions &opts, MPI_Comm comm_t_, ParMesh *pmesh)

   : MFEMBraidApp(comm_t_, opts.t_start, opts.t_final, opts.num_time_steps),
     options(opts),
     dg_fe_coll(opts.order, pmesh->Dimension(), opts.basis_type),
     h1_fe_coll(opts.order, pmesh->Dimension()),
     source(source_function),
     velocity(pmesh->Dimension(), velocity_function),
     inflow(inflow_function),
     u0(u0_function),
     // diff(-opts.diffusion), // diffusion goes in the r.h.s. with a minus
     MeshInfo(opts.max_levels)
{
   Step_calls_counter = Norm_calls_counter = 0;

   InitMultilevelApp(pmesh, opts.par_ref_levels, opts.spatial_coarsen);

   // Set the initial condition
   x[0]->ProjectCoefficient(u0);
   HypreParVector *U = x[0]->GetTrueDofs();
   SetInitialCondition(U);
   if (opts.init_rand)
   {
      int rank_t, rank_x = pmesh->GetMyRank(), glob_id;
      MPI_Comm_rank(comm_t, &rank_t);
      glob_id = rank_x + pmesh->GetNRanks()*rank_t;
      unsigned seed = 285136749 + glob_id;
      SetRandomInitVectors(seed);
   }

   SetVisHostAndPort(opts.vishost, opts.visport);
   SetVisSampling(opts.vis_time_steps, opts.vis_braid_steps);
   SetVisScreenshots(opts.vis_screenshots);
}

DGAdvectionApp::~DGAdvectionApp()
{
   for (int l = 0; l < GetNumSpaceLevels(); l++)
   {
      delete B[l];
      delete K[l];
      delete M[l];
      delete m[l];
      delete k[l];
   }
}

int DGAdvectionApp::Step(braid_Vector    u_,
                         braid_Vector    ustop_,
                         braid_Vector    fstop_,
                         BraidStepStatus &pstatus)
{
   // This contains one small change over the default Step, we store the Space-Time mesh info

   // Store Space-Time Mesh Info
   BraidVector *u     = (BraidVector*) u_;

   // BraidVector *start  = new BraidVector(0, *X0);
   // braid_Vector start2 = (braid_Vector) start;

   int level = u->level;
   double tstart, tstop, dt;
   int braid_level, iter;

   pstatus.GetTstartTstop(&tstart, &tstop);
   pstatus.GetLevel(&braid_level);
   pstatus.GetIter(&iter);
   dt = tstop - tstart;

   // Setup time-dependent source term if needed
   // !!! Is this right? I think for implicit, we should set b at time = t + dt, explicit at time = t (does it really matter?? just need to solve something...)
   if (options.ode_solver_type < 10) source.SetTime(tstart);
   else source.SetTime(tstart+dt);
   b[level]->Assemble();
   if (options.dirichlet_bcs)
   {
      ParGridFunction *u = new ParGridFunction(fe_space[level]);
      u->ProjectCoefficient(u0);
      Array<int> ess_bdr(mesh[level]->bdr_attributes.Max());
      ess_bdr = 1;
      k[level]->EliminateEssentialBC(ess_bdr, *u, *(b[level]));
      m[level]->EliminateEssentialBC(ess_bdr, *u, *(b[level]));
      delete u;
   }
   delete B[level];
   B[level] =  b[level]->ParallelAssemble();

   Step_calls_counter++;
   // Keep this for debugging:
#if 0
   cout << "\rStep #: " << setw(6) << Step_calls_counter
        << ", tstart: " << setw(12) << tstart
        << ", tstop: " << setw(12) << tstop
        << ", dt: " << setw(12) << dt
        << ", iter: " << setw(2) << iter
        << ", braid level: " << setw(2) << braid_level
        << flush;
#endif

   // This causes some kind of bug sometimes...
   // if (iter > 2)
   // {
   //    *start = 0.0;
   //    Sum(1.0, ustop_, 0.0, start2);
   //    Sum(iter*(1 - 0.1), u_, iter*0.1, start2);
   // }
   // else
   // {
   //    *start = 0.0;
   //    Sum(1.0, u_, 0.0, start2);
   // }

   MeshInfo.SetRow(braid_level, level, x[u->level]->ParFESpace()->GetParMesh(), dt);

   if ((braid_level == 0) || (options.krylov_coarse == false))
   {
       // Call the default Step
       MFEMBraidApp::Step(u_,ustop_,fstop_, pstatus);
   }
   else
   {
      double norm_u_sq = InnerProduct(u, u);
       
      if(norm_u_sq > 0.0)
      {
         // Generate Krylov Space
         Arnoldi arn(options.krylov_size*braid_level, mesh[0]->GetComm());
         // Arnoldi arn(options.krylov_size, mesh[0]->GetComm());
         arn.SetOperator(*ode[0]);
         arn.GenKrylovSpace(*u);
         Vector ubar;
         arn.ApplyVT(*u, ubar);

         // Setup the ODE solver
         ODESolver *ode_solver = NULL;
         switch (options.ode_solver_type)
         {
            case 1: ode_solver = new ForwardEulerSolver; break;
            case 2: ode_solver = new RK2Solver(1.0); break;
            case 3: ode_solver = new RK3SSPSolver; break;
            case 4: ode_solver = new RK4Solver; break;
            case 6: ode_solver = new RK6Solver; break;

            case 11: ode_solver = new BackwardEulerSolver; break;
            case 12: ode_solver = new SDIRK23Solver(2); break;
            case 13: ode_solver = new SDIRK33Solver; break;

            default: ode_solver = new RK4Solver; break;
         }
         ode_solver->Init(arn.GetH());

         // Do time-stepping
         int m;
         if( options.cfactor0 == -1)
         {
            m = pow(options.cfactor, braid_level);
         }
         else
         {
            m = options.cfactor0*pow(options.cfactor, braid_level-1);
         }

         double dt_fine = dt / ( (double) m);
         for(int k = 0; k < m; k++)
         {
            // Step() advances tstart by dt_fine
            ode_solver->Step(ubar, tstart, dt_fine);
         }

         // Convert ubar back to full space
         arn.ApplyV(ubar, *u);

         delete ode_solver;
      }

      // no refinement
      pstatus.SetRFactor(1);

   }

   return 0;
}

int DGAdvectionApp::SpatialNorm(braid_Vector  u_,
                                double       *norm_ptr)
{
   BraidVector  *u  = (BraidVector*) u_;
   FE_Evolution *ev = dynamic_cast<FE_Evolution*>(ode[u->level]);
   MFEM_ASSERT(ev, "expected object of type FE_Evolution");
   *norm_ptr = std::sqrt(ev->InnerProduct(*u, *u));
   Norm_calls_counter++;
   return 0;
}

// Allocate data structures for the given number of spatial levels. Used by
// InitMultilevelApp.
void DGAdvectionApp::AllocLevels(int num_levels)
{
   M.SetSize(num_levels, NULL);
   K.SetSize(num_levels, NULL);
   B.SetSize(num_levels, NULL);
   m.SetSize(num_levels, NULL);
   k.SetSize(num_levels, NULL);
   b.SetSize(num_levels, NULL);
}

// Construct the ParFiniteElmentSpace for the given mesh. Used by
// InitMultilevelApp.
ParFiniteElementSpace *DGAdvectionApp::ConstructFESpace(ParMesh *pmesh)
{
   if (options.problem == 0) return new ParFiniteElementSpace(pmesh, &h1_fe_coll);
   else return new ParFiniteElementSpace(pmesh, &dg_fe_coll);
}

// Assuming mesh[l] and fe_space[l] are set, initialize, ode[l], solver[l],
// and max_dt[l]. Used by InitMultilevelApp.
void DGAdvectionApp::InitLevel(int l)
{

   int myid;
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   // Create the components needed for the time-dependent operator, ode[l]:
   // the mass matrix, M; the advection (+diffusion) matrix, K; and the inflow
   // b.c. vector, B.
   const int skip_zeros = 0;
   m[l] = new ParBilinearForm(fe_space[l]);
   if (!options.lump_mass)
      m[l]->AddDomainIntegrator(new MassIntegrator);
   else
      m[l]->AddDomainIntegrator(new LumpedIntegrator(new MassIntegrator));

   b[l] = new ParLinearForm(fe_space[l]);

   if (options.problem == 0)
   {
      b[l]->AddDomainIntegrator(new DomainLFIntegrator(source));
      b[l]->Assemble();

      // Diffusion operator
      k[l] = new ParBilinearForm(fe_space[l]);
      ConstantCoefficient diff_coef(-options.diffusion);
      k[l]->AddDomainIntegrator(new DiffusionIntegrator(diff_coef));
      k[l]->Assemble();

      // Assmeble mass matrix and rhs vector
      m[l]->Assemble();

      // Set Dirichlet BCs if requested
      if (options.dirichlet_bcs)
      {
         Array<int> ess_bdr(mesh[l]->bdr_attributes.Max());
         ess_bdr = 1;
         ParGridFunction *u = new ParGridFunction(fe_space[l]);
         u->ProjectCoefficient(u0);
         k[l]->EliminateEssentialBC(ess_bdr, *u, *(b[l]));
         m[l]->EliminateEssentialBC(ess_bdr, *u, *(b[l]));
         delete u;
      }

      // Finalize and assemble matrices
      k[l]->Finalize();
      m[l]->Finalize();
      K[l] = k[l]->ParallelAssemble();
      M[l] = m[l]->ParallelAssemble();
      B[l] = b[l]->ParallelAssemble();
   }
   else
   {
      k[l] = new ParBilinearForm(fe_space[l]);
      k[l]->AddDomainIntegrator(new ConvectionIntegrator(velocity, -1.0));
      k[l]->AddInteriorFaceIntegrator( new TransposeIntegrator(new DGTraceIntegrator(velocity, 1.0, -0.5)));
      k[l]->AddBdrFaceIntegrator( new TransposeIntegrator(new DGTraceIntegrator(velocity, 1.0, -0.5)));
      ConstantCoefficient diff_coef(-options.diffusion);
      double sigma, kappa;
      sigma = -1.0;
      kappa = (options.order+1)*(options.order+1);
      k[l]->AddDomainIntegrator(new DiffusionIntegrator(diff_coef));
      k[l]->AddInteriorFaceIntegrator( new DGDiffusionIntegrator(diff_coef, sigma, kappa));
      k[l]->Assemble(skip_zeros);

      b[l] = new ParLinearForm(fe_space[l]);
      b[l]->AddDomainIntegrator(new DomainLFIntegrator(source));
      b[l]->AddBdrFaceIntegrator( new BoundaryFlowIntegrator(inflow, velocity, -1.0, -0.5));
      
      m[l]->Assemble();

      if (options.dirichlet_bcs)
      {
        ParGridFunction *u = new ParGridFunction(fe_space[l]);
        u->ProjectCoefficient(u0);
        Array<int> ess_bdr(mesh[l]->bdr_attributes.Max());
        ess_bdr = 1;        
        k[l]->EliminateEssentialBC(ess_bdr, *u, *(b[l]));
        m[l]->EliminateEssentialBC(ess_bdr, *u, *(b[l]));

        delete u;
      }

      m[l]->Finalize();
      k[l]->Finalize(skip_zeros);
      b[l]->Assemble();

      M[l] = m[l]->ParallelAssemble();
      K[l] = k[l]->ParallelAssemble();
      B[l] = b[l]->ParallelAssemble();
   }

   if (l == 0 && options.write_matrices)
   {
      if (myid == 0) cout << "\nWriting the mass and advection matrices, M and K, for"
              " level 0.\n" << endl;
      string filename = "M_prob" + to_string(options.problem) + "_size" + to_string(options.n_x);
      M[l]->Print(filename.c_str());
      filename = "K_prob" + to_string(options.problem) + "_size" + to_string(options.n_x);
      K[l]->Print(filename.c_str());
   }

   // Create the time-dependent operator, ode[l]
   FE_Evolution *fe_ev = new FE_Evolution(*M[l], *K[l], &(B[l]), options);

   fe_ev->SetPreconditionerType(options.prec_type);
   ode[l] = fe_ev;

   // Setup the ODE solver, solver[l]
   ODESolver *ode_solver = NULL;
   ARKODESolver *arkode = NULL;
   switch (options.ode_solver_type)
   {
      case 1: ode_solver = new ForwardEulerSolver; break;
      case 2: ode_solver = new RK2Solver(1.0); break;
      case 3: ode_solver = new RK3SSPSolver; break;
      case 4: ode_solver = new RK4Solver; break;
      case 6: ode_solver = new RK6Solver; break;

      case 11: ode_solver = new BackwardEulerSolver; break;
      case 12: ode_solver = new SDIRK23Solver(2); break;
      case 13: ode_solver = new SDIRK33Solver; break;

      case 21:
      {
        const double reltol = 1e-2, abstol = 1e-2;
        arkode = new ARKODESolver(ARKODESolver::IMPLICIT); 
        arkode->SetSStolerances(reltol, abstol); // What should these be???
        arkode->SetMaxStep( options.cfactor * (options.t_final - options.t_start) / options.num_time_steps ); // Is this right???
        arkode->SetERKTableNum(SDIRK_2_1_2);
        break;
      }

      default: ode_solver = new RK4Solver; break;
   }
   if (ode_solver) solver[l] = ode_solver;
   else if (arkode) solver[l] = arkode;

   // Associate the ODE operator, ode[l], with the ODE solver, solver[l]
   solver[l]->Init(*ode[l]);

   // Set max_dt[l] = 1.01*dt[0]*(2^l)
   //max_dt[l] = 1.01 * options.dt * (1 << (l+1));// start coarsening on level 2
   max_dt[l] = 1.01 * options.dt * (1 << l);      // start coarsening on level 1
}

void DGAdvectionApp::PrintStats(MPI_Comm comm)
{
   if (mesh[0]->GetMyRank() == 0)
   {
      int loc_calls[2], tot_calls[2]; // 0 - Step, 1 - Norm
      loc_calls[0] = Step_calls_counter;
      loc_calls[1] = Norm_calls_counter;
      MPI_Reduce(loc_calls, tot_calls, 2, MPI_INT, MPI_SUM, 0, comm_t);

      int my_rank_t;
      MPI_Comm_rank(comm_t, &my_rank_t);
      if (my_rank_t == 0)
      {
         cout << "\n Total number of 'Step' calls = " << tot_calls[0]
              << "\n Total number of 'Norm' calls = " << tot_calls[1] << '\n';
      }
   }

   MeshInfo.Print(comm);
}

void DGAdvectionApp::PlotMode()
{
   int myid;
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);
   if (myid == 0)
   {
      char filename[256];
      sprintf(filename,"slowMode_size%d_k%d_solver%d.0", options.n_x, options.cfactor, options.ode_solver_type);
      printf(filename);
      HYPRE_ParVector readHypreVector = hypre_ParVectorRead(mesh[0]->GetComm(), filename);

      HypreParVector mfemHypreVector(readHypreVector);

      ParGridFunction plotGridFunction(fe_space[0], mfemHypreVector);

      // Save the mesh and the solution in parallel. This output can
      // be viewed later using GLVis: "glvis -np <np> -m mesh -g sol".
      std::ostringstream mesh_name, sol_name;
      mesh_name << "modeMesh_size" << options.n_x << "." << std::setfill('0') << std::setw(6) << mesh[0]->GetMyRank();
      sol_name << "slowMode_size" << options.n_x << "_k" << options.cfactor << "_solver" << options.ode_solver_type << "." << std::setfill('0') << std::setw(6) << mesh[0]->GetMyRank();

      std::ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(8);
      mesh[0]->Print(mesh_ofs);

      std::ofstream sol_ofs(sol_name.str().c_str());
      sol_ofs.precision(8);
      plotGridFunction.Save(sol_ofs);
   }
}

// Initial condition
double source_function(Vector &x, double t)
{
   int dim = x.Size();

   // map to the reference [-1,1] domain
   Vector X(dim);
   for (int i = 0; i < dim; i++)
   {
      double center = (bb_min[i] + bb_max[i]) * 0.5;
      X(i) = 2 * (x(i) - center) / (bb_max[i] - bb_min[i]);
   }

   if (problem == 0)
   {
      switch (dim)
      {
         double r;
         case 1:
            r = min(abs(X(0) - sin(t)), 1.0);
            return exp(-1.0/(1.0 - r*r));
         case 2:
            r = min(sqrt( (X(0) - sin(t))*(X(0) - sin(t)) + (X(1) - cos(t))*(X(1) - cos(t)) ), 1.0);
            return exp(-1.0/(1.0 - r*r));
         case 3:
            r = min(sqrt( (X(0) - sin(t))*(X(0) - sin(t)) + (X(1) - cos(t))*(X(1) - cos(t)) + (X(2) - cos(t))*(X(2) - cos(t)) ), 1.0);
            return exp(-1.0/(1.0 - r*r));
      }
   }
   else
   {
      if (X(0) < -0.25 && X(0) > -0.75 && X(1) < -0.25 && X(1) > -0.75) return sin(t)*sin(t);
      else return 0.0;

   }

   return 0.0;
}

// Velocity coefficient
void velocity_function(const Vector &x, Vector &v)
{
   int dim = x.Size();

   // map to the reference [-1,1] domain
   Vector X(dim);
   for (int i = 0; i < dim; i++)
   {
      double center = (bb_min[i] + bb_max[i]) * 0.5;
      X(i) = 2 * (x(i) - center) / (bb_max[i] - bb_min[i]);
   }

    if (problem == 0)
    {
       // Zero
       switch (dim)
       {
          case 1: v(0) = 0.0; break;
          case 2: v(0) = 0.0; v(1) = 0.0; break;
          case 3: v(0) = 0.0; v(1) = 0.0; v(2) = 0.0; break;
       }
    }
    else if (problem < 10)
    {
       // Translations in 1D, 2D, and 3D
       switch (dim)
       {
          case 1: v(0) = 1.0; break;
          case 2: v(0) = sqrt(2./3.); v(1) = sqrt(1./3.); break;
          case 3: v(0) = sqrt(3./6.); v(1) = sqrt(2./6.); v(2) = sqrt(1./6.); break;
       }
    }
    else if (problem < 20)
    {
       // Clockwise rotation in 2D around the origin
       const double w = M_PI/2;
       switch (dim)
       {
          case 1: v(0) = 1.0; break;
          case 2: v(0) = w*X(1); v(1) = -w*X(0); break;
          case 3: v(0) = w*X(1); v(1) = -w*X(0); v(2) = 0.0; break;
       }
    }
    else if (problem < 30)
    {
       // Clockwise twisting rotation in 2D around the origin
       const double w = M_PI/2;
       double d = max((X(0)+1.)*(1.-X(0)),0.) * max((X(1)+1.)*(1.-X(1)),0.);
       d = d*d;
       switch (dim)
       {
          case 1: v(0) = 1.0; break;
          case 2: v(0) = d*w*X(1); v(1) = -d*w*X(0); break;
          case 3: v(0) = d*w*X(1); v(1) = -d*w*X(0); v(2) = 0.0; break;
       }
    }
}

// Initial condition
double u0_function(Vector &x)
{
   int dim = x.Size();

   // map to the reference [-1,1] domain
   Vector X(dim);
   for (int i = 0; i < dim; i++)
   {
      double center = (bb_min[i] + bb_max[i]) * 0.5;
      X(i) = 2 * (x(i) - center) / (bb_max[i] - bb_min[i]);
   }

    if (problem == 0)
    {
       switch (dim)
       {
          case 1:
             return (X(0) - 1.0)*(X(0) + 1.0);
          case 2:
             return (X(0) - 1.0)*(X(0) + 1.0)*(X(1) - 1.0)*(X(1) + 1.0);
          case 3:
             return (X(0) - 1.0)*(X(0) + 1.0)*(X(1) - 1.0)*(X(1) + 1.0)*(X(2) - 1.0)*(X(2) + 1.0);
       }
    }
    else if (problem < 10)
    {
       switch (dim)
       {
          case 1:
             return exp(-40.*pow(X(0)-0.5,2));
          case 2:
          case 3:
          {
             double rx = 0.45, ry = 0.25, cx = 0., cy = -0.2, w = 10.;
             if (dim == 3)
             {
                const double s = (1. + 0.25*cos(2*M_PI*X(2)));
                rx *= s;
                ry *= s;
             }
             return ( erfc(w*(X(0)-cx-rx))*erfc(-w*(X(0)-cx+rx)) *
                      erfc(w*(X(1)-cy-ry))*erfc(-w*(X(1)-cy+ry)) )/16;
          }
       }
    }
    else if (problem < 20)
    {
       double x_ = X(0), y_ = X(1), rho, phi;
       rho = hypot(x_, y_);
       phi = atan2(y_, x_);
       if (rho >= 1.0) return 0.0;
       else return pow(sin(M_PI*rho),2)*sin(3*phi);
    }
    else if (problem < 30)
    {
       const double f = M_PI;
       return sin(f*X(0))*sin(f*X(1));
    }
   return 0.0;
}

// Inflow boundary condition (zero for the problems considered in this example)
double inflow_function(Vector &x)
{
   switch (problem)
   {
      case 0:
      case 1:
      case 2:
      case 3: return 0.0;
   }
   return 0.0;
}
