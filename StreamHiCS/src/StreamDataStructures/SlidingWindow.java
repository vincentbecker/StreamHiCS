package StreamDataStructures;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Queue;
import java.util.Random;

import org.apache.commons.math3.util.MathArrays;

import weka.core.Instance;

/**
 * A sliding window implementation offering all the functionality to be used in
 * {@link StreamHiCS}.
 * 
 * @author Vincent
 *
 */
public class SlidingWindow extends DataStreamContainer {

	/**
	 * The window length of the {@link SlidingWindow}.
	 */
	private int windowLength;
	/**
	 * The number of {@link Instance}s currently contained in the
	 * {@link SlidingWindow}.
	 */
	private int numberOfInstances = 0;
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
	 * A generator for random numbers.
	 */
	private Random generator;
	/**
	 * A comparator to sort the internal double arrays.
	 */
	private ArrayIndexComparator indexComparator;

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
			throw new IllegalArgumentException(
					"The window length cannot be 0 or negative.");
		}
		this.windowLength = windowLength;
		this.instanceQueue = new ArrayList<Queue<Double>>(numberOfDimensions);
		// Creating all the single queues
		for (Queue<Double> q : instanceQueue) {
			// Array Deque is not thread safe!
			q = new ArrayDeque<Double>();
		}
		this.numberOfDimensions = numberOfDimensions;
		generator = new Random();
		this.indexComparator = new ArrayIndexComparator();
	}

	@Override
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

	@Override
	public int getNumberOfInstances() {
		return numberOfInstances;
	}

	@Override
	public double[] getProjectedData(int dimension) {
		Double[] temp = new Double[numberOfInstances];
		temp = instanceQueue.get(dimension).toArray(temp);
		double[] result = new double[numberOfInstances];
		// Cast all elements to double
		for (int i = 0; i < numberOfInstances; i++) {
			result[i] = (double) temp[i];
		}
		return result;
	}

	@Override
	public double[] getSlicedData(Subspace subspace, int dimension,
			double selectionAlpha) {
		// Store the indexes of all selected instances in a bit vector
		boolean[] selectedObjects = new boolean[numberOfInstances];
		for (int i = 0; i < numberOfInstances; i++) {
			selectedObjects[i] = false;
		}
		Double[] tempData = new Double[numberOfInstances];
		double[] dimData = new double[numberOfInstances];

		// Get subspace dimensions and shuffle them
		int[] dimensions = subspace.getDimensions();
		MathArrays.shuffle(dimensions);

		int selectionSize;

		for (int dim : dimensions) {
			if (dim != dimension) {
				selectionSize = (int) (countUnselected(selectedObjects) * selectionAlpha);
				// Select a random block of sorted data of size selectionSize
				// for that dimension
				tempData = instanceQueue.get(dim).toArray(tempData);
				// Cast all elements to double
				for (int i = 0; i < numberOfInstances; i++) {
					dimData[i] = (double) tempData[i];
				}
				// Sort the array indexes according to the double array
				indexComparator.setArray(dimData);
				Integer[] indexes = indexComparator.createIndexArray();
				Arrays.sort(indexes, indexComparator);

				// Select a random starting point
				if (selectionSize > numberOfInstances) {
					// Take all data
					maskIndexes(selectedObjects, indexes, 0, selectionSize);
				} else {
					// Start at a random point and take the selectionSize
					int rnd = generator.nextInt(numberOfInstances
							- selectionSize + 1);
					maskIndexes(selectedObjects, indexes, rnd, selectionSize);
				}
			}
		}

		// Get the data from dimension
		tempData = new Double[numberOfInstances];
		tempData = instanceQueue.get(dimension).toArray(tempData);
		dimData = new double[numberOfInstances];
		// Cast all elements to double
		for (int i = 0; i < numberOfInstances; i++) {
			dimData[i] = (double) tempData[i];
		}

		// Select all the data from the given dimension according to the mask
		double[] slicedData = new double[numberOfDimensions
				- countUnselected(selectedObjects)];
		int j = 0;
		for (int index = 0; index < numberOfInstances; index++) {
			if (selectedObjects[index]) {
				slicedData[j] = dimData[index];
				j++;
			}
		}

		return slicedData;
	}

	/**
	 * Masks the mask array at the indexes specified in the indexes array
	 * starting at a specific point and taking a specific amount.
	 * 
	 * @param mask
	 *            The mask array
	 * @param indexes
	 *            The array containing the indexes.
	 * @param startingPoint
	 *            The point from which index to start masking.
	 * @param selectionSize
	 *            The number of indexes which should be masked.
	 */
	private void maskIndexes(boolean[] mask, Integer[] indexes,
			int startingPoint, int selectionSize) {
		if (startingPoint < 0 || startingPoint + selectionSize > mask.length) {
			throw new IllegalArgumentException("Selection outside of range.");
		}
		for (int i = startingPoint; i < startingPoint + selectionSize; i++) {
			mask[indexes[i]] = true;
		}
	}

	/**
	 * Returns the number of false, i.e. unselected, entries in the given
	 * boolean array.
	 * 
	 * @param mask
	 *            THe boolean array
	 * @return The number of false, i.e. unselected, entries.
	 */
	private int countUnselected(boolean[] mask) {
		int count = 0;
		for (int i = 0; i < mask.length; i++) {
			if (!mask[i]) {
				count++;
			}
		}
		return count;
	}

}
