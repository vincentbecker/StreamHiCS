package pruning;

import java.util.ArrayList;
import java.util.Comparator;

import subspace.Subspace;
import subspace.SubspaceSet;

/**
 * A pruning method similar to the one of the {@link SimplePruner} but it avoids
 * the cascading pruning effect. The algorithm starts with the most
 * high-dimensional spaces and remove appropriate {@link Subspace}s. THese are
 * fully removed before continuing. Smaller {@link Subspace} will not be pruned
 * just because there are intermediate {@link Subspace}s.
 * 
 * @author Vincent
 *
 */
public class TopDownPruner extends AbstractPruner {

	/**
	 * The difference in contrast allowed to prune a {@link Subspace}, if a
	 * correlated super-space exists.
	 */
	private double pruningDifference;

	public TopDownPruner(double pruningDifference) {
		this.pruningDifference = pruningDifference;
	}

	@Override
	public SubspaceSet prune(SubspaceSet subspaceSet) {
		// Sort the subspaces set according to subspace length in descending
		// order
		subspaceSet.getSubspaces().sort(new Comparator<Subspace>() {
			@Override
			public int compare(Subspace s1, Subspace s2) {
				if (s1.size() > s2.size()) {
					return -1;
				} else if (s1.size() < s2.size()) {
					return 1;
				}
				return 0;
			}
		});

		ArrayList<Integer> discard = new ArrayList<Integer>();
		Subspace s0;
		Subspace si;

		SubspaceSet resultSet = new SubspaceSet();

		while (!subspaceSet.isEmpty() && subspaceSet.getSubspace(0).size() > 2) {
			discard.clear();
			// Get first subspace
			s0 = subspaceSet.getSubspace(0);
			discard.add(0);
			resultSet.addSubspace(s0);
			for (int i = 1; i < subspaceSet.size(); i++) {
				si = subspaceSet.getSubspace(i);
				if (si.isSubspaceOf(s0) && si.getContrast() <= (s0.getContrast() + pruningDifference)) {
					discard.add(i);
				}
			}
			// Discard
			SubspaceSet remainingSet = new SubspaceSet();
			for (int i = 0; i < subspaceSet.size(); i++) {
				if (!discard.contains(i)) {
					remainingSet.addSubspace(subspaceSet.getSubspace(i));
				}
			}
			// Continue
			subspaceSet = remainingSet;
		}

		// There might be two dimensional subspaces left over
		resultSet.addSubspaces(subspaceSet);

		return resultSet;
	}

}
