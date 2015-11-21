package pruning;

import subspace.SubspaceSet;

public abstract class AbstractPruner {

	public abstract SubspaceSet prune(SubspaceSet subspaceSet);
}
