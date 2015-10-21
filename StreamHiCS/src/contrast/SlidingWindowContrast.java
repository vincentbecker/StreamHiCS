package contrast;

import streamDataStructures.DataStreamContainer;
import streamDataStructures.SlidingWindow;
import streamDataStructures.Subspace;
import weka.core.Instance;

public class SlidingWindowContrast extends Contrast {
	/**
	 * The number of {@link Instance} that are observed before the
	 * {@link Subspace} contrasts are checked again.
	 */
	private int updateInterval;
	/**
	 * To count the number of {@link Instance}s observed since the last
	 * {@link Subspace} evaluation.
	 */
	private int currentCount = 0;
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
	public SlidingWindowContrast(Callback callback, int numberOfDimensions, int updateInterval, int m, double alpha, int windowLength) {
		super(callback, m, alpha);
		this.updateInterval = updateInterval;
		// dataStreamContainer = new SelfOrganizingMap(numberOfDimensions, 100);
		dataStreamContainer = new SlidingWindow(numberOfDimensions, windowLength);
	}

	@Override
	public void add(Instance instance) {
		dataStreamContainer.add(instance);
		currentCount++;
		if (currentCount >= updateInterval) {
			onAlarm();
			currentCount = 0;
		}
	}

	/**
	 * Clears all stored @link{Instance}s.
	 */
	public void clear() {
		dataStreamContainer.clear();
		currentCount = 0;
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
	public double[] getProjectedData(int referenceDimension) {
		return dataStreamContainer.getProjectedData(referenceDimension);
	}

	@Override
	public double[] getSlicedData(int[] shuffledDimensions, double selectionAlpha) {
		return dataStreamContainer.getSlicedData(shuffledDimensions, selectionAlpha);
	}

	@Override
	public int getNumberOfElements() {
		return dataStreamContainer.getNumberOfInstances();
	}
}
