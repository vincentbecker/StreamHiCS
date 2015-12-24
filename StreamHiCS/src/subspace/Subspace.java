package subspace;

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
	 * The contrast of the subspace at the time it was last evaluated.
	 */
	private double contrast = 0;

	/**
	 * Returns the contrast value-
	 * 
	 * @return The contrast value:
	 */
	public double getContrast() {
		return contrast;
	}

	/**
	 * Sets the contrast attribute to the given value.
	 * 
	 * @param contrast
	 *            The contrast value.
	 */
	public void setContrast(double contrast) {
		this.contrast = contrast;
	}

	/**
	 * Create a {@link Subspace} object.
	 */
	public Subspace() {
		dimensions = new ArrayList<Integer>();
	}

	/**
	 * Creates a {@link Subspace} object and adds the given dimensions.
	 * 
	 * @param dimensions
	 *            A list of dimensions of variable length.
	 */
	public Subspace(int... dimensions) {
		this.dimensions = new ArrayList<Integer>();
		for (int i = 0; i < dimensions.length; i++) {
			addDimension(dimensions[i]);
		}
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
	public int size() {
		return dimensions.size();
	}

	/**
	 * Checks whether this {@link Subspace} contains the given dimension.
	 * 
	 * @param dimension
	 *            The dimension
	 * @return True, if the dimension is contained, false otherwise.
	 */
	public boolean contains(int dimension) {
		return dimensions.contains(dimension);
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

	/**
	 * Discards a dimension at the given index from this {@link Subspace}.
	 * 
	 * @param index
	 *            The index of the dimension.
	 */
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
		for (int i = 0; i < this.size(); i++) {
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
	@Override
	public boolean equals(Object s2) {
		if (!(s2 instanceof Subspace)) {
			return false;
		}
		if (this == s2) {
			return true;
		}
		Subspace subspaceToTest = (Subspace) s2;
		if (this.size() != subspaceToTest.size()) {
			return false;
		}
		for (int dimension : dimensions) {
			if (!subspaceToTest.dimensions.contains(dimension)) {
				return false;
			}
		}
		return true;
	}

	/**
	 * Merges the two given {@link Subspace}s of the same length k > 0, if they
	 * have the first k-1 elements in common. The dimensions in the merged
	 * subspace are sorted in ascending order.
	 * 
	 * 
	 */
	public static Subspace merge(Subspace s1, Subspace s2) {
		int k = s1.size();
		if (k == 0) {
			return null;
		}
		if (s2.size() != k) {
			return null;
		}
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
	 * Merges the two given {@link Subspace}s of the same length k > 0 and if
	 * they have any k - 1 elements in common, no matter in which order.
	 * 
	 * @param s1
	 *            The first {@link Subspace}.
	 * @param s2
	 *            The second {@link Subspace}.
	 * @return The reference to a new subspace object, if the two
	 *         {@link Subspace}s could be merged, null otherwise.
	 */
	public static Subspace mergeFull(Subspace s1, Subspace s2) {
		int k = s1.size();
		if (k == 0) {
			return null;
		}
		if (s2.size() != k) {
			return null;
		}
		int counter = 0;
		Subspace s = s2.copy();
		int dimension;
		for (int i = 0; i < k; i++) {
			dimension = s1.getDimension(i);
			if (s2.contains(dimension)) {
				counter++;
			} else {
				s.addDimension(dimension);
			}
		}

		if (counter != k - 1) {
			return null;
		}

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
	 * Checks whether this {@link Subspace} is a subspace of another subspace,
	 * meaning that the other subspace contains every dimension, this subspace
	 * contains.
	 * 
	 * @param s
	 *            The other {@link Subspace}.
	 * @return True, if this subspace is a subspace of the other subspace, false
	 *         otherwise.
	 */
	public boolean isSubspaceOf(Subspace s) {
		for (int i = 0; i < dimensions.size(); i++) {
			if (!s.dimensions.contains(this.dimensions.get(i))) {
				return false;
			}
		}
		return true;
	}

	/**
	 * Checks, whether the {@link Subspace} is empty, i.e. contains no
	 * dimensions.
	 * 
	 * @return True, if the subspace is empty, false otherwise.
	 */
	public boolean isEmpty() {
		return dimensions.isEmpty();
	}

	/**
	 * Returns a string representation of this object.
	 * 
	 * @return A string representation of this object.
	 */
	public String toString() {
		return dimensions.toString();
	}

	/**
	 * Returns the number of common dimensions with the other {@link Subspace}.
	 * 
	 * @param s2
	 *            The other subspace
	 * @return The number of common dimensions.
	 */
	public int cut(Subspace s2) {
		int count = 0;
		for (Integer dim : dimensions) {
			if (s2.contains(dim)) {
				count++;
			}
		}
		return count;
	}
}
