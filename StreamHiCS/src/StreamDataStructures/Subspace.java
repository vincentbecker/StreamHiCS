package streamDataStructures;

import java.util.ArrayList;

/**
 * This class represents a subspace consisting of the dimensions it contains.
 * 
 * @author Vincent
 *
 */
public class Subspace {

	/**
	 * The dimensions the {@link Subspace} consists of.
	 */
	private ArrayList<Integer> dimensions;

	/**
	 * Create a {@link Subspace} object.
	 */
	public Subspace() {
		dimensions = new ArrayList<Integer>();
	}

	/**
	 * Adds the specified dimension to the {@link Subspace}. If the dimension is
	 * already contained in the {@link Subspace} nothing is changed.
	 * 
	 * @param dimension
	 *            The dimension to be added.
	 */
	public void addDimension(int dimension) {
		if (!dimensions.contains(dimension)) {
			dimensions.add(dimension);
		}
	}

	/**
	 * Returns the number of dimensions in the {@link Subspace}.
	 * 
	 * @return The number of dimensions in the {@link Subspace}.
	 */
	public int getSize() {
		return dimensions.size();
	}

	/**
	 * Returns the dimension at the specified index.
	 * 
	 * @param index
	 *            The index.
	 * @return The dimension at the specified index.
	 */
	public int getDimension(int index) {
		return (int) dimensions.get(index);
	}

	/**
	 * The dimensions of the {@link Subspace} as an int array.
	 * 
	 * @return An int array containing the dimensions.
	 */
	public int[] getDimensions() {
		int numberOfDimensions = dimensions.size();
		Integer[] temp = new Integer[numberOfDimensions];
		temp = dimensions.toArray(temp);
		int[] result = new int[numberOfDimensions];
		// Cast all elements to int
		for (int i = 0; i < numberOfDimensions; i++) {
			result[i] = (int) temp[i];
		}
		return result;
	}

	public void discardDimension(int index) {
		dimensions.remove(index);
	}

	/**
	 * Carries out a deep copy of a {@link Subspace} object.
	 * 
	 * @return The reference to the copy.
	 */
	public Subspace copy() {
		Subspace c = new Subspace();
		for (int i = 0; i < this.getSize(); i++) {
			c.dimensions.add(this.dimensions.get(i));
		}
		return c;
	}

	/**
	 * Tests if two {@link Subspace}s are equal.
	 * 
	 * @param s2
	 *            The other {@link Subspace}.
	 * @return
	 */
	public boolean equals(Subspace s2) {
		if (this.getSize() != s2.getSize()) {
			return false;
		}
		for (int dimension : dimensions) {
			if (!s2.dimensions.contains(dimension)) {
				return false;
			}
		}
		return true;
	}

	/**
	 * Merges the two given {@link Subspace}s of the same length k, if they have
	 * the first k-1 elements in common. The dimensions in the merged subspace
	 * are sorted in ascending order.
	 * 
	 * @param s1
	 *            The first {@link Subspace}.
	 * @param s2
	 *            The second {@link Subspace}.
	 * @return The reference to a new subspace object, if the two
	 *         {@link Subspace}s could be merged, null otherwise.
	 */
	public static Subspace merge(Subspace s1, Subspace s2) {
		int k = s1.getSize();
		for (int i = 0; i < k - 1; i++) {
			if (s1.getDimension(i) != s2.getDimension(i)) {
				return null;
			}
		}
		Subspace s = s1.copy();
		s.addDimension(s2.getDimension(k - 1));
		s.sort();
		return s;
	}

	/**
	 * Sorts the dimensions in an ascending order.
	 */
	public void sort() {
		dimensions.sort(new IntegerComparator());
	}

	/**
	 * Returns a string representation of this object.
	 * 
	 * @return A string representation of this object.
	 */
	public String toString() {
		return dimensions.toString();
	}
}
