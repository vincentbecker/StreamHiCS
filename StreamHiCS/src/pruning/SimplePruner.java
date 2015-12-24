package pruning;

import java.util.ArrayList;

import subspace.Subspace;
import subspace.SubspaceSet;

/**
 * Uses the simplest form of pruning. Iterates over the {@link SubspaceSet} and
 * deletes {@link Subspace}s which have a super-space with a contrast greater
 * then contrast(subspace) - pruningDifference. This may cause cascading pruning
 * problems, since a small subspace with a high contrast is pruned by a very
 * high-dimensional {@link Subspace} with low contrast because there are
 * intermediate {@link Subspace}.
 * 
 * @author Vincent
 *
 */
public class SimplePruner extends AbstractPruner {

	/**
	 * The difference in contrast allowed to prune a {@link Subspace}, if a
	 * correlated super-space exists.
	 */
	private double pruningDifference;

	/**
	 * Creates an instance of this class.
	 * 
	 * @param pruningDifference
	 *            The difference in contrast allowed to prune a {@link Subspace}
	 *            , if a correlated super-space exists
	 */
	public SimplePruner(double pruningDifference) {
		this.pruningDifference = pruningDifference;
	}

	/**
	 * If a {@link Subspace} is a subspace of another subspace with a higher
	 * contrast value, then it is discarded.
	 */
	@Override
	public SubspaceSet prune(SubspaceSet subspaceSet) {
		ArrayList<Integer> discard = new ArrayList<Integer>();
		int l = subspaceSet.size();
		Subspace si;
		Subspace sj;
		boolean discarded;
		for (int i = 0; i < l; i++) {
			discarded = false;
			si = subspaceSet.getSubspace(i);
			for (int j = 0; j < l && !discarded; j++) {
				if (i != j) {
					sj = subspaceSet.getSubspace(j);
					// If the correlated subspace contains a superset that has
					// at least (nearly) the same contrast we discard the
					// current subspace
					if (si.isSubspaceOf(sj) && si.getContrast() <= (sj.getContrast() + pruningDifference)) {
						discard.add(i);
						discarded = true;
					}
				}
			}
		}
		SubspaceSet prunedSet = new SubspaceSet();
		for (int i = 0; i < l; i++) {
			if (!discard.contains(i)) {
				prunedSet.addSubspace(subspaceSet.getSubspace(i));
			}
		}

		return prunedSet;

	}

}
