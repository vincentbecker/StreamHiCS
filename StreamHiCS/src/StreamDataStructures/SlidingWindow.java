package streamdatastructures;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Queue;

import weka.core.Instance;

/**
 * A sliding window implementation offering all the functionality to be used in
 * {@link StreamHiCS}.
 * 
 * @author Vincent
 *
 */
public class SlidingWindow {

	/**
	 * The number of instances currently contained in the {@link SlidingWindow}.
	 */
	private int numberOfInstances;

	/**
	 * The window length of the {@link SlidingWindow}.
	 */
	private int windowLength;

	/**
	 * An {@link ArrayList} containing one queue for {@link Double}s for each
	 * dimension. An instance is stored by storing each attribute value in the
	 * corresponding queue.
	 */
	private ArrayList<Queue<Double>> instanceQueue;

	/**
	 * The number of dimensions of the full space.
	 */
	private int numberOfDimensions;

	/**
	 * Creates a {@link SlidingWindow} object with the specified window length.
	 * 
	 * @param numberOfDimensions
	 *            The dimensionality of the space.
	 * @param windowLength
	 *            The window length of the {@link SlidingWindow}, i.e. the
	 *            number of {@link Instance}s the window holds before it starts
	 *            discarding older {@link Instance}s.
	 */
	public SlidingWindow(int numberOfDimensions, int windowLength) {
		if (windowLength <= 0) {
			throw new IllegalArgumentException("The window length cannot be 0 or negative.");
		}
		this.windowLength = windowLength;
		this.instanceQueue = new ArrayList<Queue<Double>>(numberOfDimensions);
		// Creating all the single queues
		for (int i = 0; i < numberOfDimensions; i++) {
			// Array Deque is not thread safe!
			instanceQueue.add(new ArrayDeque<Double>());
		}
		this.numberOfDimensions = numberOfDimensions;
	}

	/**
	 * Adds an {@link Instance} to this {@link SlidingWindow}. If the window was
	 * full, the oldest instance is removed to provide space for the new
	 * {@link Instance}.
	 * 
	 * @param instance
	 *            The {@link Instance} to be added.
	 */
	public void add(Instance instance) {
		if (numberOfInstances >= windowLength) {
			// Removing the oldest instance
			for (Queue<Double> q : instanceQueue) {
				q.remove();
			}
			numberOfInstances--;
		}
		// Adding the instance
		for (int i = 0; i < numberOfDimensions; i++) {
			instanceQueue.get(i).add(instance.value(i));
		}
		numberOfInstances++;
	}

	/**
	 * Clear the window.
	 */
	public void clear() {
		for (Queue<Double> q : instanceQueue) {
			q.clear();
		}
		numberOfInstances = 0;
	}

	/**
	 * Returns the number of {@link Instance} currently contained in this
	 * {@link SlidingWindow}.
	 * 
	 * @return The number of instance currently contained in the window.
	 */
	public int getNumberOfInstances() {
		return numberOfInstances;
	}

	/**
	 * Returns the one dimensional data from the given dimension.
	 * 
	 * @param dimension
	 *            The dimension the data should be taken from
	 * @return The data from the given dimension.
	 */
	public double[] getDimensionData(int dimension) {
		double[] data = new double[numberOfInstances];
		Double[] tempData = new Double[numberOfInstances];
		tempData = instanceQueue.get(dimension).toArray(tempData);
		for (int i = 0; i < numberOfInstances; i++) {
			data[i] = tempData[i];
		}
		return data;
	}
}
