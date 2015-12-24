package pruning;

import subspace.SubspaceSet;

/**
 * This class represents a pruning procedure which receives a
 * {@link SubspaceSet} and carries out the pruning.
 * 
 * @author Vincent
 *
 */
public abstract class AbstractPruner {

	/**
	 * Carries out the pruning procedure specified in the subclasses.
	 * 
	 * @param subspaceSet
	 *            The {@link SubspaceSet} to nbe pruned.
	 * @return The pruned {@link SubspaceSet}.
	 */
	public abstract SubspaceSet prune(SubspaceSet subspaceSet);
}
