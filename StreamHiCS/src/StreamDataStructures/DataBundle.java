package streamdatastructures;

import org.apache.commons.math3.util.MathArrays;

/**
 * This class represents a structure to hold one-dimensional data and
 * corresponding weights.
 * 
 * @author Vincent
 *
 */
public class DataBundle {

	/**
	 * The one dimensional data.
	 */
	private double[] data;

	/**
	 * The weights.
	 */
	private double[] weights;

	/**
	 * The indexes after sorting.
	 */
	private double[] sortedIndexes;

	/**
	 * The sorted data.
	 */
	private double[] sortedData;

	/**
	 * The weights after sorting.
	 */
	private double[] sortedWeights;

	/**
	 * Creates an instance of this class.
	 * 
	 * @param data
	 *            The none dimensional data
	 * @param weights
	 *            The weights corresponding to the data
	 */
	public DataBundle(double[] data, double[] weights) {
		int n = data.length;
		if (n != weights.length) {
			throw new IllegalArgumentException("Data and weights have different length.");
		}
		this.data = data;
		this.weights = weights;
	}

	/**
	 * Returns the data.
	 * 
	 * @return The data.
	 */
	public double[] getData() {
		return data;
	}

	/**
	 * Returns the weights.
	 * 
	 * @return The weights.
	 */
	public double[] getWeights() {
		return weights;
	}

	/**
	 * Returns the indexes which are sorted according to the data, if the data
	 * was sorted.
	 * 
	 * @return The indexes which are sorted according to the data, if the data
	 *         was sorted.
	 */
	public double[] getSortedIndexes() {
		return sortedIndexes;
	}

	/**
	 * Returns the sorted data, if the data was sorted.
	 * 
	 * @return The sorted data, if the data was sorted.
	 */
	public double[] getSortedData() {
		return sortedData;
	}

	/**
	 * Returns the weights which are sorted according to the data, if the data
	 * was sorted.
	 * 
	 * @return The weights which are sorted according to the data, if the data
	 *         was sorted.
	 */
	public double[] getSortedWeights() {
		return sortedWeights;
	}

	/**
	 * Returns the size of the data.
	 * 
	 * @return The size of the data.
	 */
	public int size() {
		return data.length;
	}

	/**
	 * Checks, whether this {@link DataBundle} is empty.
	 * 
	 * @return True, if empty, false otherwise.
	 */
	public boolean isEmpty() {
		if (data == null || data.length == 0) {
			return true;
		}
		return false;
	}

	/**
	 * Sorts the data and holds the result in an extra array. The indexes and
	 * weights sorted according to the data are stored as well.
	 */
	public void sort() {
		int n = data.length;
		this.sortedIndexes = new double[n];
		this.sortedData = new double[n];
		this.sortedWeights = new double[n];

		// Copying the data
		for (int i = 0; i < n; i++) {
			sortedIndexes[i] = i;
			sortedData[i] = data[i];
			sortedWeights[i] = weights[i];
		}
		// Sorting
		MathArrays.sortInPlace(sortedData, sortedIndexes, sortedWeights);
	}
}
