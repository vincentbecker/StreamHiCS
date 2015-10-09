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
	
	public abstract StatisticsBundle getProjectedDataStaistics(int referenceDimension);
	
	public abstract StatisticsBundle getSlicedDataStaistics(int[] shuffledDimensions);
}
