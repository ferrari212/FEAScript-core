/**
 * ════════════════════════════════════════════════════════════════
 *  FEAScript Core Library
 *  Lightweight Finite Element Simulation in JavaScript
 *  Version: 0.3.0 (RC) | https://feascript.com
 *  MIT License © 2023–2026 FEAScript
 * ════════════════════════════════════════════════════════════════
 */

// Internal imports
import { initializeFEA, performIsoparametricMapping1D } from "../mesh/meshUtils.js";
import { GenericBoundaryConditions } from "./genericBoundaryConditions.js";
import { basicLog, debugLog, errorLog } from "../utilities/logging.js";

/**
 * Assembles the discrete weak form of a general 1D scalar linear PDE of the form
 *
 *   -d/dx (A(x) du/dx) + B(x) du/dx + C(x) u = D(x)
 *
 * where:
 *   - u(x) is the scalar unknown field being solved for,
 *   - A(x) is the diffusion or stiffness coefficient,
 *   - B(x) is the advection or transport coefficient,
 *   - C(x) is the reaction or damping coefficient,
 *   - D(x) is the source or forcing term.
 *
 * This is a general-purpose finite-element model rather than a fluid-only
 * formulation. In practice, the same structure can represent diffusion,
 * transport, reaction, damping, or stiffness-dominated scalar field problems.
 *
 * In the finite-element implementation, the residual and Jacobian are assembled
 * from the Galerkin weak form at each Gauss point. The code evaluates the
 * coefficient functions A(x), B(x), C(x), and D(x) at the mapped physical
 * coordinate x and accumulates the corresponding diffusion, advection,
 * reaction, and source terms into the global system.
 *
 * The resulting algebraic system is:
 *
 *   K(u) = F
 *
 * where K is the assembled Jacobian matrix and F is the residual forcing term.
 *
 * @param {object} meshData - Prepared mesh data containing nodal coordinates,
 *                           element connectivity, element order, and dimension.
 * @param {object} boundaryConditions - Boundary condition definition object.
 * @param {object} coefficientFunctions - Functions A(x), B(x), C(x), D(x)
 *                                        describing the PDE coefficients.
 * @returns {{ jacobianMatrix: number[][], residualVector: number[] }}
 *          The assembled global Jacobian matrix and residual vector.
 */
export function assembleGeneralFormPDEMat(meshData, boundaryConditions, coefficientFunctions) {
  basicLog("Starting general form PDE matrix assembly...");

  // Extract mesh data
  const {
    nodesXCoordinates,
    nodesYCoordinates,
    nop,
    boundaryElements,
    totalElements,
    meshDimension,
    elementOrder,
  } = meshData;

  // Extract coefficient functions
  const { A, B, C, D } = coefficientFunctions;

  // Initialize FEA components
  const FEAData = initializeFEA(meshData);
  const {
    residualVector,
    jacobianMatrix,
    localToGlobalMap,
    basisFunctions,
    gaussPoints,
    gaussWeights,
    nodesPerElement,
  } = FEAData;

  if (meshDimension === "1D") {
    // 1D general form PDE

    // Matrix assembly
    for (let elementIndex = 0; elementIndex < totalElements; elementIndex++) {
      // Map local element nodes to global mesh nodes
      for (let localNodeIndex = 0; localNodeIndex < nodesPerElement; localNodeIndex++) {
        // Convert to 0-based indexing
        localToGlobalMap[localNodeIndex] = Math.abs(nop[elementIndex][localNodeIndex]) - 1;
      }

      // Loop over Gauss points
      for (let gaussPointIndex = 0; gaussPointIndex < gaussPoints.length; gaussPointIndex++) {
        // Get basis functions for the current Gauss point
        const { basisFunction, basisFunctionDerivKsi } = basisFunctions.getBasisFunctions(
          gaussPoints[gaussPointIndex],
        );

        // Perform isoparametric mapping
        const { detJacobian, basisFunctionDerivX } = performIsoparametricMapping1D({
          basisFunction,
          basisFunctionDerivKsi,
          nodesXCoordinates,
          localToGlobalMap,
          nodesPerElement,
        });

        // Calculate the physical coordinate for this Gauss point
        let xCoord = 0;
        for (let i = 0; i < nodesPerElement; i++) {
          xCoord += nodesXCoordinates[localToGlobalMap[i]] * basisFunction[i];
        }

        // Evaluate coefficient functions at this physical coordinate
        const a = A(xCoord);
        const b = B(xCoord);
        const c = C(xCoord);
        const d = D(xCoord);

        // Computation of Galerkin's residuals and local Jacobian matrix
        for (let localNodeIndex1 = 0; localNodeIndex1 < nodesPerElement; localNodeIndex1++) {
          const globalNodeIndex1 = localToGlobalMap[localNodeIndex1];

          // Source term contribution to residual vector
          residualVector[globalNodeIndex1] -=
            gaussWeights[gaussPointIndex] * detJacobian * d * basisFunction[localNodeIndex1];

          for (let localNodeIndex2 = 0; localNodeIndex2 < nodesPerElement; localNodeIndex2++) {
            const globalNodeIndex2 = localToGlobalMap[localNodeIndex2];

            // Diffusion term
            jacobianMatrix[globalNodeIndex1][globalNodeIndex2] +=
              gaussWeights[gaussPointIndex] *
              detJacobian *
              a *
              basisFunctionDerivX[localNodeIndex1] *
              basisFunctionDerivX[localNodeIndex2];

            // Advection term
            jacobianMatrix[globalNodeIndex1][globalNodeIndex2] -=
              gaussWeights[gaussPointIndex] *
              detJacobian *
              b *
              basisFunctionDerivX[localNodeIndex2] *
              basisFunction[localNodeIndex1];

            // Reaction term
            jacobianMatrix[globalNodeIndex1][globalNodeIndex2] -=
              gaussWeights[gaussPointIndex] *
              detJacobian *
              c *
              basisFunction[localNodeIndex1] *
              basisFunction[localNodeIndex2];
          }
        }
      }
    }
  } else if (meshDimension === "2D") {
    errorLog("2D general form PDE is not yet supported in assembleGeneralFormPDEMat.");
    // 2D general form PDE - empty for now
  }

  // Apply boundary conditions
  const genericBoundaryConditions = new GenericBoundaryConditions(
    boundaryConditions,
    boundaryElements,
    nop,
    meshDimension,
    elementOrder,
  );

  // Apply Dirichlet boundary conditions only
  genericBoundaryConditions.imposeDirichletBoundaryConditions(residualVector, jacobianMatrix);

  basicLog("General form PDE matrix assembly completed");

  return {
    jacobianMatrix,
    residualVector,
  };
}

/**
 * Assembles the local element contribution for the same general 1D PDE model
 * used by the frontal solver. The element residual and local Jacobian are
 * formed by evaluating the same diffusion, advection, and reaction terms at
 * each Gauss point, but only for the current element rather than the full mesh.
 *
 * The returned local operators are later assembled into the global system by
 * the frontal solver. This allows the same PDE structure to be used in both
 * the full global assembly path and the element-by-element frontal assembly path.
 *
 * @param {object} data - Element assembly input for the frontal solver, containing:
 *   - elementIndex: index of the current element in the mesh,
 *   - nop: element connectivity array,
 *   - meshData: mesh coordinates and dimensionality metadata,
 *   - basisFunctions: interpolation and derivative providers for the element,
 *   - FEAData: numerical integration data such as Gauss points, weights,
 *              and nodes-per-element,
 *   - coefficientFunctions: callback functions A(x), B(x), C(x), and D(x).
 * @returns {{ localJacobianMatrix: number[][], localResidualVector: number[], ngl: number[] }}
 *          Local Jacobian matrix, local residual vector, and global node list.
 */
export function assembleGeneralFormPDEFront({
  elementIndex,
  nop,
  meshData,
  basisFunctions,
  FEAData,
  coefficientFunctions,
}) {
  // Extract numerical integration parameters and mesh coordinates
  const { gaussPoints, gaussWeights, nodesPerElement } = FEAData;
  const { nodesXCoordinates, nodesYCoordinates, meshDimension } = meshData;
  const { A, B, C, D } = coefficientFunctions;

  // Initialize local Jacobian matrix and local residual vector
  const localJacobianMatrix = Array(nodesPerElement)
    .fill()
    .map(() => Array(nodesPerElement).fill(0));
  const localResidualVector = Array(nodesPerElement).fill(0);

  // Build the mapping from local node indices to global node indices
  const ngl = Array(nodesPerElement);
  const localToGlobalMap = Array(nodesPerElement);
  for (let localNodeIndex = 0; localNodeIndex < nodesPerElement; localNodeIndex++) {
    ngl[localNodeIndex] = Math.abs(nop[elementIndex][localNodeIndex]);
    localToGlobalMap[localNodeIndex] = Math.abs(nop[elementIndex][localNodeIndex]) - 1;
  }

  if (meshDimension === "1D") {
    // 1D general form PDE

    // Loop over Gauss points
    for (let gaussPointIndex = 0; gaussPointIndex < gaussPoints.length; gaussPointIndex++) {
      // Get basis functions for the current Gauss point
      const { basisFunction, basisFunctionDerivKsi } = basisFunctions.getBasisFunctions(
        gaussPoints[gaussPointIndex],
      );

      // Perform isoparametric mapping
      const { detJacobian, basisFunctionDerivX } = performIsoparametricMapping1D({
        basisFunction,
        basisFunctionDerivKsi,
        nodesXCoordinates,
        localToGlobalMap,
        nodesPerElement,
      });

      // Calculate the physical coordinate for this Gauss point
      let xCoord = 0;
      for (let i = 0; i < nodesPerElement; i++) {
        xCoord += nodesXCoordinates[localToGlobalMap[i]] * basisFunction[i];
      }

      // Evaluate coefficient functions at this physical coordinate
      const a = A(xCoord);
      const b = B(xCoord);
      const c = C(xCoord);
      const d = D(xCoord);

      // Computation of local Jacobian matrix and residual vector
      for (let localNodeIndex1 = 0; localNodeIndex1 < nodesPerElement; localNodeIndex1++) {
        // Source term contribution to local residual vector
        localResidualVector[localNodeIndex1] -=
          gaussWeights[gaussPointIndex] * detJacobian * d * basisFunction[localNodeIndex1];

        for (let localNodeIndex2 = 0; localNodeIndex2 < nodesPerElement; localNodeIndex2++) {
          // Diffusion term
          localJacobianMatrix[localNodeIndex1][localNodeIndex2] +=
            gaussWeights[gaussPointIndex] *
            detJacobian *
            a *
            basisFunctionDerivX[localNodeIndex1] *
            basisFunctionDerivX[localNodeIndex2];

          // Advection term
          localJacobianMatrix[localNodeIndex1][localNodeIndex2] -=
            gaussWeights[gaussPointIndex] *
            detJacobian *
            b *
            basisFunctionDerivX[localNodeIndex2] *
            basisFunction[localNodeIndex1];

          // Reaction term
          localJacobianMatrix[localNodeIndex1][localNodeIndex2] -=
            gaussWeights[gaussPointIndex] *
            detJacobian *
            c *
            basisFunction[localNodeIndex1] *
            basisFunction[localNodeIndex2];
        }
      }
    }
  } else if (meshDimension === "2D") {
    errorLog("2D general form PDE is not yet supported in assembleGeneralFormPDEFront.");
    // 2D general form PDE - empty for now
  }

  return {
    localJacobianMatrix,
    localResidualVector,
    ngl,
  };
}
