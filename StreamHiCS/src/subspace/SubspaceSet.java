package subspace;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;

import org.apache.commons.math3.util.MathArrays;

/**
 * This class represents a set of {@link Subspace}s.
 * 
 * @author Vincent
 *
 */
public class SubspaceSet {

	/**
	 * The internal set of {@link Subspace}s.
	 */
	private ArrayList<Subspace> subspaces;

	/**
	 * Returns the subspaces.
	 * 
	 * @return The {@link ArrayList} containing the {@link Subspace}s.
	 */
	public ArrayList<Subspace> getSubspaces() {
		return subspaces;
	}

	/**
	 * Creates a {@link SubspaceSet} object.
	 */
	public SubspaceSet() {
		subspaces = new ArrayList<Subspace>();
	}

	/**
	 * Returns the {@link Subspace} at the given index.
	 * 
	 * @param index
	 *            The index.
	 * @return The {@link Subspace at the given index.}
	 */
	public Subspace getSubspace(int index) {
		return subspaces.get(index);
	}

	/**
	 * Checks whether the set is empty.
	 * 
	 * @return True, if the set is empty, false otherwise.
	 */
	public boolean isEmpty() {
		return subspaces.isEmpty();
	}

	/**
	 * Returns the number of {@link Subspace}s contained in the set.
	 * 
	 * @return The number of {@link Subspace}s contained in the set.
	 */
	public int size() {
		return subspaces.size();
	}

	/**
	 * Adds a {@link Subspace} to this set. If the subspace is already contained
	 * nothing is done.
	 * 
	 * @param subspace
	 *            The {@link Subspace} to be added.
	 */
	public void addSubspace(Subspace subspace) {
		if (!subspaces.contains(subspace)) {
			subspaces.add(subspace);
		}
	}

	/**
	 * Adds all subspaces from the given {@link SubspaceSet} to this set.
	 * 
	 * @param set2
	 *            The other {@link SubspaceSet}.
	 */
	public void addSubspaces(SubspaceSet set2) {
		for (Subspace s : set2.getSubspaces()) {
			addSubspace(s);
		}
	}

	/**
	 * Removes the given subspace from this set, if it is contained.
	 * 
	 * @param subspace
	 *            The subspace to be removed.
	 */
	public void removeSubspace(Subspace subspace) {
		subspaces.remove(subspace);
	}

	/**
	 * Clears the set.
	 */
	public void clear() {
		subspaces.clear();
	}

	/**
	 * Keeps the {@link Subspace}s with the highest contrast values (cutoff
	 * many) from this set and discards the others.
	 */
	public void selectTopK(int cutoff) {
		int l = subspaces.size();
		double[] contrasts = new double[l];
		double[] indexes = new double[l];
		for (int i = 0; i < l; i++) {
			contrasts[i] = subspaces.get(i).getContrast();
			indexes[i] = i;
		}
		// Sort the array and the indexes accordingly
		MathArrays.sortInPlace(contrasts, MathArrays.OrderDirection.DECREASING, indexes);
		ArrayList<Subspace> prunedSubspaces = new ArrayList<Subspace>();
		for (int i = 0; i < l && i < cutoff; i++) {
			prunedSubspaces.add(subspaces.get((int) indexes[i]));
		}
		subspaces = prunedSubspaces;
	}

	/**
	 * Check if this set and another set are equal, meaning they contain the
	 * same {@link Subspace}s.
	 * 
	 * @param set2
	 *            The other {@link SubspaceSet}.
	 * @return True, if equal, false otherwise.
	 */
	@Override
	public boolean equals(Object set2) {
		if (!(set2 instanceof SubspaceSet)) {
			return false;
		}
		if (this == set2) {
			return true;
		}
		SubspaceSet setToTest = (SubspaceSet) set2;
		if (this.size() != setToTest.size()) {
			return false;
		}
		for (Subspace s : subspaces) {
			if (!setToTest.subspaces.contains(s)) {
				return false;
			}
		}
		return true;
	}

	/**
	 * Checks if this set contains the given {@link Subspace}.
	 * 
	 * @param s
	 *            The subspace
	 * @return True, if the {@link Subspace} is contained, false otherwise.
	 */
	public boolean contains(Subspace s) {
		return subspaces.contains(s);
	}

	/**
	 * Sorts the {@link Subspace}s contained in this set. Assumes that the
	 * subspaces themselves are sorted.
	 */
	public void sort() {
		Collections.sort(subspaces, new Comparator<Subspace>() {

			@Override
			public int compare(Subspace s1, Subspace s2) {
				int s1Size = s1.size();
				int s2Size = s2.size();
				int l = Math.min(s1Size, s2Size);
				int s1Dim = 0;
				int s2Dim = 0;
				for (int i = 0; i < l; i++) {
					s1Dim = s1.getDimension(i);
					s2Dim = s2.getDimension(i);
					if (s1Dim < s2Dim) {
						return -1;
					} else if (s1Dim > s2Dim) {
						return 1;
					}
				}
				if (s1Size < s2Size) {
					return -1;
				} else if (s1Size > s2Size) {
					return 1;
				} else {
					return 0;
				}
			}
		});
	}

	/**
	 * Returns a string representation of this object.
	 * 
	 * @return A string representation of this object.
	 */
	@Override
	public String toString() {
		String rep = "";
		for (Subspace s : subspaces) {
			rep += (s.toString() + " ");
		}
		return rep;
	}
}
