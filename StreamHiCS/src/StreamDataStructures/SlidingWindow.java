package streamDataStructures;

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
		for (int i = 0; i < numberOfDimensions; i++) {
			// Array Deque is not thread safe!
			instanceQueue.add(new ArrayDeque<Double>());
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
		// Get subspace dimensions and shuffle them
		int[] dimensions = subspace.getDimensions();
		MathArrays.shuffle(dimensions);

		double[] dimData;
		ArrayList<Integer> selectedIndexes = new ArrayList<Integer>(
				numberOfInstances);
		// Fill the list with all the indexes
		for (int i = 0; i < numberOfInstances; i++) {
			selectedIndexes.add(i);
		}
		int selectionSize = (int) (numberOfInstances * selectionAlpha);

		for (int dim : dimensions) {
			if (dim != dimension) {
				// Get all the data for the specific dimension that is selected
				dimData = getSelectedData(dim, selectedIndexes);
				indexComparator.setArray(dimData);
				Arrays.sort(selectedIndexes, indexComparator);

				if (selectionSize > numberOfInstances) {
					// Take all data
					selectIndexes(selectedIndexes, 0, selectionSize);
				} else {
					// Start at a random point and take the selectionSize
					int rnd = generator.nextInt(numberOfInstances
							- selectionSize + 1);
					selectIndexes(selectedIndexes, rnd, selectionSize);
				}

				selectionSize = (int) (selectedIndexes.size() * selectionAlpha);
			}
		}

		// Get the selected data from dimension
		return getSelectedData(dimension, selectedIndexes);
	}

	private double[] getSelectedData(int dimension,
			ArrayList<Integer> selectedIndexes) {
		double[] data = new double[selectedIndexes.size()];
		Double[] tempData = new Double[numberOfInstances];
		tempData = instanceQueue.get(dimension).toArray(tempData);
		for (int i = 1; i < selectedIndexes.size(); i++) {
			data[i] = tempData[selectedIndexes.get(i)];
		}
		return data;
	}

	private void selectIndexes(ArrayList<Integer> indexes, int startingPoint,
			int selectionSize) {
		if (startingPoint < 0 || startingPoint + selectionSize > indexes.size()) {
			throw new IllegalArgumentException("Selection outside of range.");
		}
		// Keep all the indexes within the range
		for (int i = 0; i < indexes.size(); i++) {
			if (!(startingPoint <= i && i < startingPoint + selectionSize)) {
				indexes.remove(i);
			}
		}
	}
}
