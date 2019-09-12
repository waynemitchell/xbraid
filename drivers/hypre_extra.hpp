// Copyright (c) 2013, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory. Written by
// Jacob Schroder, Rob Falgout, Tzanio Kolev, Ulrike Yang, Veselin
// Dobrev, et al. LLNL-CODE-660355. All rights reserved.
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
//

#ifndef braid_hypre_extra_HEADER
#define braid_hypre_extra_HEADER

#include "_hypre_parcsr_mv.h"
#include "_braid.h"

// Additional "HYPRE" functions
namespace hypre
{
   /**------------------------------------------------------------------------
    * Perform the operation A += beta*B, assuming that the sparsity pattern of
    * A contains that of B.
    * -----------------------------------------------------------------------*/
   HYPRE_Int
   hypre_CSRMatrixSum( hypre_CSRMatrix *A,
                       HYPRE_Complex    beta,
                       hypre_CSRMatrix *B )
   {
      HYPRE_Complex    *A_data   = hypre_CSRMatrixData(A);
      HYPRE_Int        *A_i      = hypre_CSRMatrixI(A);
      HYPRE_Int        *A_j      = hypre_CSRMatrixJ(A);
      HYPRE_Int         nrows_A  = hypre_CSRMatrixNumRows(A);
      HYPRE_Int         ncols_A  = hypre_CSRMatrixNumCols(A);
      HYPRE_Complex    *B_data   = hypre_CSRMatrixData(B);
      HYPRE_Int        *B_i      = hypre_CSRMatrixI(B);
      HYPRE_Int        *B_j      = hypre_CSRMatrixJ(B);
      HYPRE_Int         nrows_B  = hypre_CSRMatrixNumRows(B);
      HYPRE_Int         ncols_B  = hypre_CSRMatrixNumCols(B);

      HYPRE_Int         ia, j, pos;
      HYPRE_Int        *marker;

      if (nrows_A != nrows_B || ncols_A != ncols_B)
      {
         hypre_printf("Warning! incompatible matrix dimensions!\n");
         return -1;
      }

      marker = _braid_CTAlloc(HYPRE_Int, ncols_A);
      for (ia = 0; ia < ncols_A; ia++)
         marker[ia] = -1;

      for (ia = 0; ia < nrows_A; ia++)
      {
         for (j = A_i[ia]; j < A_i[ia+1]; j++)
            marker[A_j[j]] = j;

         for (j = B_i[ia]; j < B_i[ia+1]; j++)
         {
            pos = marker[B_j[j]];
            if (pos < A_i[ia])
            {
               hypre_printf("Warning! hypre_CSRMatrixSum: Incorrect input!\n");
               return -2;
            }
            A_data[pos] += beta * B_data[j];
         }
      }

      _braid_TFree(marker);
      return 0;
   }

   /**------------------------------------------------------------------------
    * Return a new matrix containing the sum of A and B, assuming that both
    * matrices use the same row and column partitions and the same col_map_offd
    * arrays.
    * -----------------------------------------------------------------------*/

hypre_ParCSRMatrix *
hypre_ParCSRMatrixAdd(hypre_ParCSRMatrix *A,
                      hypre_ParCSRMatrix *B)
{
   MPI_Comm            comm   = hypre_ParCSRMatrixComm(A);
   hypre_CSRMatrix    *A_diag = hypre_ParCSRMatrixDiag(A);
   hypre_CSRMatrix    *A_offd = hypre_ParCSRMatrixOffd(A);
   HYPRE_Int          *A_cmap = hypre_ParCSRMatrixColMapOffd(A);
   HYPRE_Int           A_cmap_size = hypre_CSRMatrixNumCols(A_offd);
   hypre_CSRMatrix    *B_diag = hypre_ParCSRMatrixDiag(B);
   hypre_CSRMatrix    *B_offd = hypre_ParCSRMatrixOffd(B);
   HYPRE_Int          *B_cmap = hypre_ParCSRMatrixColMapOffd(B);
   HYPRE_Int           B_cmap_size = hypre_CSRMatrixNumCols(B_offd);
   hypre_ParCSRMatrix *C;
   hypre_CSRMatrix    *C_diag;
   hypre_CSRMatrix    *C_offd;
   HYPRE_Int          *C_cmap;
   HYPRE_Int           im;
   HYPRE_Int           cmap_differ;

   /* Check if A_cmap and B_cmap are the same. */
   cmap_differ = 0;
   if (A_cmap_size != B_cmap_size)
   {
      cmap_differ = 1; /* A and B have different cmap_size */
   }
   else
   {
      for (im = 0; im < A_cmap_size; im++)
      {
         if (A_cmap[im] != B_cmap[im])
         {
            cmap_differ = 1; /* A and B have different cmap arrays */
            break;
         }
      }
   }

   if ( cmap_differ == 0 )
   {
      /* A and B have the same column mapping for their off-diagonal blocks so
         we can sum the diagonal and off-diagonal blocks separately and reduce
         temporary memory usage. */

      /* Add diagonals, off-diagonals, copy cmap. */
      C_diag = hypre_CSRMatrixAdd(A_diag, B_diag);
      if (!C_diag)
      {
         return NULL; /* error: A_diag and B_diag have different dimensions */
      }
      C_offd = hypre_CSRMatrixAdd(A_offd, B_offd);
      if (!C_offd)
      {
         hypre_CSRMatrixDestroy(C_diag);
         return NULL; /* error: A_offd and B_offd have different dimensions */
      }
      /* copy A_cmap -> C_cmap */
      C_cmap = hypre_TAlloc(HYPRE_Int, A_cmap_size, HYPRE_MEMORY_HOST);
      for (im = 0; im < A_cmap_size; im++)
      {
         C_cmap[im] = A_cmap[im];
      }

      C = hypre_ParCSRMatrixCreate(comm,
                                   hypre_ParCSRMatrixGlobalNumRows(A),
                                   hypre_ParCSRMatrixGlobalNumCols(A),
                                   hypre_ParCSRMatrixRowStarts(A),
                                   hypre_ParCSRMatrixColStarts(A),
                                   hypre_CSRMatrixNumCols(C_offd),
                                   hypre_CSRMatrixNumNonzeros(C_diag),
                                   hypre_CSRMatrixNumNonzeros(C_offd));

      /* In C, destroy diag/offd (allocated by Create) and replace them with
      C_diag/C_offd. */
      hypre_CSRMatrixDestroy(hypre_ParCSRMatrixDiag(C));
      hypre_CSRMatrixDestroy(hypre_ParCSRMatrixOffd(C));
      hypre_ParCSRMatrixDiag(C) = C_diag;
      hypre_ParCSRMatrixOffd(C) = C_offd;

      hypre_ParCSRMatrixColMapOffd(C) = C_cmap;
   }
   else
   {
      /* A and B have different column mappings for their off-diagonal blocks so
      we need to use the column maps to create full-width CSR matricies. */

      int  ierr = 0;
      hypre_CSRMatrix * csr_A;
      hypre_CSRMatrix * csr_B;
      hypre_CSRMatrix * csr_C_temp;

      /* merge diag and off-diag portions of A */
      csr_A = hypre_MergeDiagAndOffd(A);

      /* merge diag and off-diag portions of B */
      csr_B = hypre_MergeDiagAndOffd(B);

      /* add A and B */
      csr_C_temp = hypre_CSRMatrixAdd(csr_A,csr_B);

      /* delete CSR versions of A and B */
      ierr += hypre_CSRMatrixDestroy(csr_A);
      ierr += hypre_CSRMatrixDestroy(csr_B);

      /* create a new empty ParCSR matrix to contain the sum */
      C = hypre_ParCSRMatrixCreate(hypre_ParCSRMatrixComm(A),
                                   hypre_ParCSRMatrixGlobalNumRows(A),
                                   hypre_ParCSRMatrixGlobalNumCols(A),
                                   hypre_ParCSRMatrixRowStarts(A),
                                   hypre_ParCSRMatrixColStarts(A),
                                   0, 0, 0);

      /* split C into diag and off-diag portions */
      /* TODO: GenerateDiagAndOffd() uses an int array of size equal to the
         number of columns in csr_C_temp which is the global number of columns
         in A and B. This does not scale well. */
      ierr += GenerateDiagAndOffd(csr_C_temp, C,
                                  hypre_ParCSRMatrixFirstColDiag(A),
                                  hypre_ParCSRMatrixLastColDiag(A));

      /* delete CSR version of C */
      ierr += hypre_CSRMatrixDestroy(csr_C_temp);

      MFEM_VERIFY(ierr == 0, "");
   }

   /* hypre_ParCSRMatrixSetNumNonzeros(A); */

   /* Make sure that the first entry in each row is the diagonal one. */
   hypre_CSRMatrixReorder(hypre_ParCSRMatrixDiag(C));

   /* C owns diag, offd, and cmap. */
   hypre_ParCSRMatrixSetDataOwner(C, 1);
   /* C does not own row and column starts. */
   hypre_ParCSRMatrixSetRowStartsOwner(C, 0);
   hypre_ParCSRMatrixSetColStartsOwner(C, 0);

   return C;
}

   /**------------------------------------------------------------------------
    * Perform the operation A += beta*B, assuming that both matrices use the
    * same row and column partitions and the same col_map_offd arrays. Also,
    * it is assumed that the sparsity pattern of A contains that of B.
    * -----------------------------------------------------------------------*/
   HYPRE_Int
   hypre_ParCSRMatrixSum( hypre_ParCSRMatrix *A,
                          HYPRE_Complex       beta,
                          hypre_ParCSRMatrix *B )
   {
      hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(A);
      hypre_CSRMatrix *A_offd = hypre_ParCSRMatrixOffd(A);
      hypre_CSRMatrix *B_diag = hypre_ParCSRMatrixDiag(B);
      hypre_CSRMatrix *B_offd = hypre_ParCSRMatrixOffd(B);
      HYPRE_Int error;

      error = hypre_CSRMatrixSum(A_diag, beta, B_diag);
      if (!error)
         error = hypre_CSRMatrixSum(A_offd, beta, B_offd);

      return error;
   }

   HYPRE_Int
   hypre_CSRMatrixSetConstantValues( hypre_CSRMatrix *A,
                                     HYPRE_Complex    value )
   {
      HYPRE_Complex *A_data = hypre_CSRMatrixData(A);
      HYPRE_Int      A_nnz  = hypre_CSRMatrixNumNonzeros(A);
      HYPRE_Int      ia;

      for (ia = 0; ia < A_nnz; ia++)
         A_data[ia] = value;

      return 0;
   }

   HYPRE_Int
   hypre_ParCSRMatrixSetConstantValues( hypre_ParCSRMatrix *A,
                                        HYPRE_Complex       value )
   {
      hypre_CSRMatrixSetConstantValues(hypre_ParCSRMatrixDiag(A), value);
      hypre_CSRMatrixSetConstantValues(hypre_ParCSRMatrixOffd(A), value);

      return 0;
   }
}

#endif // braid_hypre_extra_HEADER
