package centroids;

import statisticalTests.StatisticsBundle;
import weka.core.Instance;

public abstract class CentroidsContainer {

	public abstract void add(Instance instance);

	/**
	 * Clears all stored data.
	 */
	public abstract void clear();

	/**
	 * Returns the number of {@link Instance}s currently contained.
	 * 
	 * @return The number of {@link Instance}s currently contained.
	 */
	public abstract int getNumberOfInstances();

	public abstract double[] getProjectedData(int referenceDimension);

	public abstract double[] getSlicedData(int[] shuffledDimensions, double selectionAlpha);

	public StatisticsBundle calculateStatistics(Centroid[] centroidSelection, int referenceDimension) {
		int totalCount = 0;
		double totalSum = 0;
		int count = 0;
		double mean = 0;
		double variance = 0;

		// Calculation of mean
		for (Centroid centroid : centroidSelection) {
			if (centroid != null) {
				count = centroid.getCount();
				totalSum += centroid.getVector()[referenceDimension] * count;
				totalCount += count;
			}
		}
		mean = totalSum / totalCount;

		// Calculation of variance
		totalSum = 0;
		for (Centroid centroid : centroidSelection) {
			if (centroid != null) {
				totalSum += centroid.getCount() * Math.pow(centroid.getVector()[referenceDimension] - mean, 2);
			}
		}
		variance = totalSum / (totalCount - 1);

		return new StatisticsBundle(mean, variance);
	}

	public abstract void densityCheck();

	public double euclideanDistance(double[] v1, double[] v2) {
		if (v1.length != v2.length) {
			throw new IllegalArgumentException("Centroid vectors are of different length.");
		}
		double distance = 0;
		for (int i = 0; i < v1.length; i++) {
			distance += Math.pow(v1[i] - v2[i], 2);
		}

		return Math.sqrt(distance);
	}
}
