package streamdatastructures;

import coreset.BucketManager;
import coreset.Point;
import moa.clusterers.streamkm.MTRandom;
import weka.core.Instance;

/**
 * This class represents an adapter to access data from a Coreset implementation
 * and directly holds the implementation.
 * 
 * @author Vincent
 *
 */
public class CoresetAdapter extends SummarisationAdapter {

	/**
	 * The {@link BucketManager} creating the coresets.
	 */
	private BucketManager bucketManager;

	/**
	 * A flag showing whether the implementation was already initialised or not.
	 */
	private boolean initialized = false;

	/**
	 * The size of the stream.
	 */
	private int width = 0;

	/**
	 * The maximum size of the coreset.
	 */
	private int coresetSize = 0;

	/**
	 * The number of {@link Instance}s seen.
	 */
	private int numberInstances = 0;

	/**
	 * The number of dimensions.
	 */
	private int d = 0;

	/**
	 * The current coreset.
	 */
	private Point[] currentCoreset;

	/**
	 * Creates an instance of this class.
	 * 
	 * @param width
	 *            The size of the stream
	 * @param coresetSize
	 *            The maximum size of a coreset.
	 */
	public CoresetAdapter(int width, int coresetSize) {
		this.width = width;
		this.coresetSize = coresetSize;
	}

	@Override
	public void addImpl(Instance instance) {
		if (!initialized) {
			d = instance.numAttributes();
			bucketManager = new BucketManager(this.width, instance.numAttributes(), this.coresetSize, new MTRandom(1));
			initialized = true;
		}

		bucketManager.insertPoint(new Point(instance, this.numberInstances));
		numberInstances++;
		currentCoreset = null;
	}

	@Override
	public void clearImpl() {
		initialized = false;
		currentCoreset = null;
		numberInstances = 0;
	}

	@Override
	public DataBundle[] getData() {
		if (currentCoreset == null) {
			currentCoreset = bucketManager.getCoresetFromManager(d);
		}

		int n = currentCoreset.length;
		if (n > 0) {
			double[][] points = new double[n][];
			double[] weights = new double[n];
			Point p;
			for (int i = 0; i < n; i++) {
				p = currentCoreset[i];
				points[i] = p.getCoordinates();
				weights[i] = p.getWeight();
			}
			// Construct the DataBundles from the data
			DataBundle[] data = new DataBundle[d];

			for (int dim = 0; dim < d; dim++) {
				double[] dimData = new double[n];
				double[] w = new double[n];
				for (int i = 0; i < n; i++) {
					dimData[i] = points[i][dim];
					// Copying weights, necessary since manipulated later
					w[i] = weights[i];
				}

				data[dim] = new DataBundle(dimData, w);
			}

			return data;

		} else {
			return null;
		}
	}

	@Override
	public int getNumberOfElements() {
		if (currentCoreset == null) {
			currentCoreset = bucketManager.getCoresetFromManager(d);
		}
		if (currentCoreset == null) {
			return 0;
		}

		return currentCoreset.length;
	}

}
