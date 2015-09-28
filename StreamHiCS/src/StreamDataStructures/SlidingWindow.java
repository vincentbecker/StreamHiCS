package StreamDataStructures;

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
		for (Queue<Double> q : instanceQueue) {
			// Array Deque is not thread safe!
			q = new ArrayDeque<Double>();
		}
		this.numberOfDimensions = numberOfDimensions;
	}

	@Override
	public void add(Instance instance) {
		if (numberOfInstances >= windowLength) {
			//Removing the oldest instance
			for(Queue<Double> q : instanceQueue){
				q.remove();
			}
			numberOfInstances--;
		}
		//Adding the instance
		for(int i = 0; i < numberOfDimensions; i++){
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
		for(int i = 0; i < numberOfInstances; i++){
			result[i] = (double) temp[i];
		}
		return result;
	}

	@Override
	public double[] getSlicedData(Subspace subspace, int dimension,
			int selectionSize) {
		// TODO Auto-generated method stub
		return null;
	}

}
