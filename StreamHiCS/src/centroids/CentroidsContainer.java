package centroids;

import contrast.DataBundle;
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
	
	public abstract Centroid[] getCentroids();

	public abstract DataBundle getProjectedData(int referenceDimension);

	public abstract DataBundle getSlicedData(int[] shuffledDimensions, double selectionAlpha);

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

		return new StatisticsBundle(mean, variance, totalCount);
	}
}
