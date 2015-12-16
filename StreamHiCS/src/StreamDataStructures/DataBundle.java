package streamdatastructures;

import org.apache.commons.math3.util.MathArrays;

public class DataBundle {
	private double[] data;
	private double[] weights;
	private double[] sortedIndexes;
	private double[] sortedData;
	private double[] sortedWeights;

	public DataBundle(double[] data, double[] weights) {
		int n = data.length;
		if (n != weights.length) {
			throw new IllegalArgumentException("Data and weights have different length.");
		}
		this.data = data;
		this.weights = weights;
	}

	public double[] getData() {
		return data;
	}

	public double[] getWeights() {
		return weights;
	}

	public double[] getSortedIndexes() {
		return sortedIndexes;
	}

	public double[] getSortedData() {
		return sortedData;
	}

	public double[] getSortedWeights() {
		return sortedWeights;
	}

	public int size() {
		return data.length;
	}

	public boolean isEmpty() {
		if (data == null || data.length == 0) {
			return true;
		}
		return false;
	}

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
