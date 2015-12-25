package subspacebuilder;

import subspace.Subspace;
import subspace.SubspaceSet;

/**
 * This class represents an abstract subspace build-up algorithm.
 * 
 * @author Vincent
 *
 */
public abstract class SubspaceBuilder {

	/**
	 * Build the correlated {@link Subspace}s.
	 * 
	 * @return A {@link SubspaceSet} containing the correlated {@link Subspace}
	 *         s.
	 */
	public abstract SubspaceSet buildCorrelatedSubspaces();

}
