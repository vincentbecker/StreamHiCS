package streamDataStructures;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Queue;

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
		Selection selectedIndexes = new Selection(numberOfInstances,
				selectionAlpha);
		// Fill the list with all the indexes
		selectedIndexes.fillRange();

		for (int dim : dimensions) {
			if (dim != dimension) {
				// Get all the data for the specific dimension that is selected
				dimData = getSelectedData(dim, selectedIndexes);
				// Reduce the number of indexes according to a new selection in
				// the current dimension
				selectedIndexes.select(dimData);
			}
		}

		// Get the selected data from dimension
		return getSelectedData(dimension, selectedIndexes);
	}

	/**
	 * Returns the data stored in this {@link SlidingWindow} corresponding to
	 * the given dimension and the specified indexes.
	 * 
	 * @param dimension
	 *            The dimension the data is taken from
	 * @param selectedIndexes
	 *            The indexes of the data point which are selected.
	 * @return The data stored in this {@link SlidingWindow} corresponding to
	 *         the given dimension and the specified indexes.
	 */
	private double[] getSelectedData(int dimension, Selection selectedIndexes) {
		double[] data = new double[selectedIndexes.size()];
		Double[] tempData = new Double[numberOfInstances];
		tempData = instanceQueue.get(dimension).toArray(tempData);
		for (int i = 0; i < selectedIndexes.size(); i++) {
			data[i] = tempData[selectedIndexes.getIndex(i)];
		}
		return data;
	}
}
