package contrast;

import streamDataStructures.DataStreamContainer;
import streamDataStructures.SlidingWindow;
import subspace.Subspace;
import weka.core.Instance;

public class SlidingWindowContrast extends Contrast {
	/**
	 * Data structure holding the {@link Instance}s.
	 */
	private DataStreamContainer dataStreamContainer;

	/**
	 * Creates a {@link SlidingWindowContrast} object with the specified update
	 * interval.
	 * 
	 * @param numberOfDimensions
	 *            The number of dimensions of the full space.
	 * @param updateInterval
	 *            The number how many {@link Instance}s are observed between
	 *            evaluations of the correlated {@link Subspace}s.
	 */
	public SlidingWindowContrast(int numberOfDimensions, int m, double alpha, int windowLength) {
		super(m, alpha);
		dataStreamContainer = new SlidingWindow(numberOfDimensions, windowLength);
	}

	@Override
	public void add(Instance instance) {
		dataStreamContainer.add(instance);
	}

	/**
	 * Clears all stored @link{Instance}s.
	 */
	public void clear() {
		dataStreamContainer.clear();
	}

	/**
	 * Returns the number of {@link Instance}s currently contained in this
	 * object.
	 * 
	 * @return The number of {@link Instance}s currently contained in this
	 *         object.
	 */
	public int getNumberOfInstances() {
		return dataStreamContainer.getNumberOfInstances();
	}

	@Override
	public DataBundle getProjectedData(int referenceDimension) {
		return dataStreamContainer.getProjectedData(referenceDimension);
	}

	@Override
	public DataBundle getSlicedData(int[] shuffledDimensions, double selectionAlpha) {
		return dataStreamContainer.getSlicedData(shuffledDimensions, selectionAlpha);
	}

	@Override
	public int getNumberOfElements() {
		return dataStreamContainer.getNumberOfInstances();
	}

	@Override
	public double[][] getUnderlyingPoints() {
		return dataStreamContainer.getUnderlyingPoints();
	}
}
