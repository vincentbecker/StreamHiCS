package streamDataStructures;

import java.util.ArrayList;

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
	 * Returns a string representation of this object.
	 * 
	 * @return A string representation of this object.
	 */
	public String toString() {
		String rep = "";
		for (Subspace s : subspaces) {
			rep += (s.toString() + " ");
		}
		return rep;
	}
}
