package streamdatastructures;

import weka.core.Instance;

/**
 * This class represents an adapter to access data from a {@link SlidingWindow}s
 * implementation.
 * 
 * @author Vincent
 *
 */
public class SlidingWindowAdapter extends SummarisationAdapter {
	
	/**
	 * The {@link SlidingWindow} instance. 
	 */
	private SlidingWindow slidingWindow;
	
	/**
	 * The number of dimensions. 
	 */
	private int d;
	
	/**
	 * Creates an instance of this class. 
	 * 
	 * @param numberOfDimensions The number of dimensions
	 * @param windowLength The size of the sliding window
	 */
	public SlidingWindowAdapter(int numberOfDimensions, int windowLength) {
		slidingWindow = new SlidingWindow(numberOfDimensions, windowLength);
		this.d = numberOfDimensions;
	}
	
	@Override
	public void addImpl(Instance instance) {
		slidingWindow.add(instance);
	}

	@Override
	public void clearImpl() {
		slidingWindow.clear();
	}

	@Override
	public DataBundle[] getData() {

		int n = slidingWindow.getNumberOfInstances();
		if (n > 0) {
			// Construct the DataBundles from the data
			DataBundle[] data = new DataBundle[d];

			for (int dim = 0; dim < d; dim++) {
				double[] w = new double[n];
				for (int i = 0; i < n; i++) {
					// All the weights are 1
					w[i] = 1;
				}

				data[dim] = new DataBundle(slidingWindow.getDimensionData(dim), w);
			}

			return data;
		} else {
			return null;
		}
	}

	@Override
	public int getNumberOfElements() {
		return slidingWindow.getNumberOfInstances();
	}

}
